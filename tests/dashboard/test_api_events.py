import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.agent.memory_db import MemoryContext
from nanobot.agent.tools.cron import CronTool
from nanobot.api.dashboard import create_dashboard_app
from nanobot.bus.queue import MessageBus
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.observability.events import DashboardEvents
from nanobot.observability.store import ObservabilityStore
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.tests.memory_test_utils import TestAgentLoop as AgentLoop


@pytest.mark.asyncio
async def test_two_websockets_follow_real_turn_and_reconnect_reads_rest(tmp_path):
    events = DashboardEvents()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (10, "test")
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done"))
    loop = AgentLoop(MessageBus(), provider, tmp_path, context_window_tokens=100000,
                     observability_events=events)
    loop.memory_service.prepare_context = AsyncMock(return_value=MemoryContext())
    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock(
        side_effect=lambda session, memory_context, **kwargs: memory_context)
    store = loop.observability_store
    store.initialize()
    app = create_dashboard_app(loop, CronService(tmp_path / "jobs.json"), store, events=events)
    try:
        async with TestClient(TestServer(app)) as client:
            first = await client.ws_connect("/api/dashboard/events")
            second = await client.ws_connect("/api/dashboard/events")
            assert (await loop.process_direct("hello", session_key="cli:test")).content == "done"

            async def finish(ws):
                seen = []
                while True:
                    event = await asyncio.wait_for(ws.receive_json(), timeout=3)
                    seen.append(event)
                    if event["type"] == "run.finished":
                        return seen

            left, right = await asyncio.gather(finish(first), finish(second))
            for received in (left, right):
                assert received[0]["type"] == "run.started"
                assert received[-1]["type"] == "run.finished"
                assert {item["run_id"] for item in received} == {left[0]["run_id"]}
            assert (await (await client.get(f"/api/dashboard/runs/{left[0]['run_id']}")).json())["run"]["status"] == "completed"
            await first.close()
            reconnected = await client.ws_connect("/api/dashboard/events")
            page = await (await client.get("/api/dashboard/runs")).json()
            assert page["items"][0]["run_id"] == left[0]["run_id"]
            await reconnected.close()
            await second.close()
    finally:
        await loop.close_mcp()


@pytest.mark.asyncio
async def test_cron_sources_publish_only_after_persist(tmp_path):
    events = DashboardEvents()
    path = tmp_path / "jobs.json"
    cron = CronService(path, on_job=lambda _: asyncio.sleep(0),
                       on_change=lambda job_id: events.publish("task.changed", job_id=job_id))
    store = ObservabilityStore(tmp_path / "dashboard.db")
    store.initialize()
    app = create_dashboard_app(object(), cron, store, events=events)
    tool = CronTool(cron)
    tool.set_context("cli", "test")
    async with TestClient(TestServer(app)) as client:
        ws = await client.ws_connect("/api/dashboard/events")
        created = await tool.execute(action="add", message="Do work", every_seconds=3600)
        job = cron.list_jobs()[0]
        assert job.id in created
        assert await ws.receive_json() == {"type": "task.changed", "job_id": job.id}
        assert path.exists()

        response = await client.patch(f"/api/dashboard/tasks/{job.id}", json={"enabled": False})
        assert response.status == 200
        assert await ws.receive_json() == {"type": "task.changed", "job_id": job.id}
        assert cron.get_job(job.id).state.next_run_at_ms is None

        cron.enable_job(job.id, True)
        assert await ws.receive_json() == {"type": "task.changed", "job_id": job.id}
        await cron.run_job(job.id)
        assert await ws.receive_json() == {"type": "task.changed", "job_id": job.id}
        assert cron.get_job(job.id).state.run_history

        one_shot = cron.add_job("Once", CronSchedule(kind="at", at_ms=4102444800000),
                                "once", delete_after_run=True)
        assert await ws.receive_json() == {"type": "task.changed", "job_id": one_shot.id}
        await cron.run_job(one_shot.id)
        assert await ws.receive_json() == {"type": "task.changed", "job_id": one_shot.id}
        assert cron.get_job(one_shot.id) is None

        cron.get_job(job.id).state.next_run_at_ms = 1
        await cron._on_timer()
        assert await ws.receive_json() == {"type": "task.changed", "job_id": job.id}
        assert cron.get_job(job.id).state.run_history

        assert job.id in await tool.execute(action="remove", job_id=job.id)
        assert await ws.receive_json() == {"type": "task.changed", "job_id": job.id}
        await ws.close()


@pytest.mark.asyncio
async def test_websocket_origin_and_shutdown(tmp_path):
    events = DashboardEvents()
    store = ObservabilityStore(tmp_path / "dashboard.db")
    store.initialize()
    app = create_dashboard_app(object(), object(), store, events=events)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/api/dashboard/events", headers={"Origin": "http://evil.test",
                                                             "Upgrade": "websocket"})
        assert response.status == 403
        ws = await client.ws_connect("/api/dashboard/events")
        await client.server.close()
        assert ws.closed or (await ws.receive()).type.name in {"CLOSE", "CLOSED", "CLOSING"}
