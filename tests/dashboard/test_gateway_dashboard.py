import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from nanobot.cli.commands import _run_gateway_services, _start_dashboard_listener
from nanobot.agent.loop import AgentLoop
from nanobot.api.dashboard import AGENT_KEY, CRON_KEY
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, KnowledgeConfig
from nanobot.cron.service import CronService
from nanobot.observability.store import ObservabilityStore
from nanobot.providers.base import GenerationSettings


@pytest.mark.asyncio
async def test_listener_uses_existing_agent_and_cron_and_closes(tmp_path, monkeypatch):
    config = Config()
    config.dashboard.port = 0  # site chooses an available test port
    store = ObservabilityStore(tmp_path / "dashboard.db")
    agent = SimpleNamespace(observability_store=store)
    cron = object()
    seen = {}
    from nanobot.api import dashboard
    real_factory = dashboard.create_dashboard_app

    def factory(agent_arg, cron_arg, store_arg, **kwargs):
        seen.update(agent=agent_arg, cron=cron_arg, store=store_arg)
        return real_factory(agent_arg, cron_arg, store_arg, **kwargs)

    monkeypatch.setattr(dashboard, "create_dashboard_app", factory)
    runner = await _start_dashboard_listener(agent, cron, config)
    assert runner is not None
    assert seen == {"agent": agent, "cron": cron, "store": store}
    site = next(iter(runner.sites))
    port = site._server.sockets[0].getsockname()[1]
    from aiohttp import ClientSession
    async with ClientSession() as client:
        async with client.get(f"http://127.0.0.1:{port}/api/dashboard/runs") as response:
            assert response.status == 200
    await runner.cleanup()
    assert not runner.sites


@pytest.mark.asyncio
async def test_fresh_workspace_binds_real_agent_memory_knowledge_and_cron(tmp_path, monkeypatch):
    from aiohttp import ClientSession
    from nanobot.agent import knowledge, memory_service
    from nanobot.api import dashboard

    class FixedEmbedding:
        dimension = 1024

        def __init__(self):
            self.query_calls = []

        async def embed_query(self, text):
            self.query_calls.append(text)
            return [0.0] * self.dimension

        async def embed_documents(self, texts):
            raise AssertionError("Fresh stores must not embed documents")

    embedder = FixedEmbedding()
    monkeypatch.setattr(memory_service, "DashScopeEmbeddingBackend", lambda **_: embedder)
    monkeypatch.setattr(knowledge, "DashScopeEmbeddingBackend", lambda **_: embedder)
    monkeypatch.setattr(knowledge, "DashScopeRerankerBackend", lambda **_: object())
    config = Config()
    config.dashboard.port = 0
    config.knowledge = KnowledgeConfig(enabled=True)
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    cron = CronService(tmp_path / "cron" / "jobs.json")
    agent = AgentLoop(MessageBus(), provider, tmp_path, cron_service=cron,
                      knowledge_config=config.knowledge)
    assert agent.memory_service.database.get_schema_version() == 8
    assert agent.web_knowledge_service.db.get_schema_version() == 4
    assert agent.knowledge_retriever is not None
    real_factory = dashboard.create_dashboard_app
    seen = {}

    def factory(agent_arg, cron_arg, store_arg, **kwargs):
        app = real_factory(agent_arg, cron_arg, store_arg, **kwargs)
        seen.update(agent=app[AGENT_KEY], cron=app[CRON_KEY])
        return app

    monkeypatch.setattr(dashboard, "create_dashboard_app", factory)
    runner = None
    try:
        runner = await _start_dashboard_listener(agent, cron, config)
        assert runner is not None
        assert seen == {"agent": agent, "cron": cron}
        port = next(iter(runner.sites))._server.sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        async with ClientSession() as client:
            for path in ("overview", "runs", "memory", "tasks"):
                async with client.get(f"{base}/api/dashboard/{path}") as response:
                    assert response.status == 200, path
                    if path == "memory":
                        data = await response.json()
                        assert data["revision"] == 0
                        assert data["records"] == []
            async with client.post(f"{base}/api/dashboard/memory/search",
                                   json={"query": "test"}) as response:
                assert response.status == 200
                assert (await response.json())["dynamic"] == []
        assert embedder.query_calls == []
    finally:
        if runner is not None:
            await runner.cleanup()
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", port)) != 0
        await agent.close_mcp()


@pytest.mark.asyncio
async def test_port_conflict_does_not_raise(tmp_path):
    config = Config()
    config.dashboard.port = 0
    store = ObservabilityStore(tmp_path / "dashboard.db")
    agent = SimpleNamespace(observability_store=store)
    first = await _start_dashboard_listener(agent, object(), config)
    assert first is not None
    config.dashboard.port = next(iter(first.sites))._server.sockets[0].getsockname()[1]
    second = await _start_dashboard_listener(agent, object(), config)
    assert second is None
    await first.cleanup()


@pytest.mark.asyncio
async def test_gateway_shutdown_drains_active_turn_and_listener(tmp_path, monkeypatch):
    config = Config()
    config.dashboard.port = 0
    store = ObservabilityStore(tmp_path / "dashboard.db")
    active = asyncio.create_task(asyncio.sleep(60))
    agent = SimpleNamespace(observability_store=store, _active_tasks={"cli:test": [active]},
                            run=AsyncMock(), stop=Mock(), close_mcp=AsyncMock())
    cron = SimpleNamespace(start=AsyncMock(), stop=Mock())
    heartbeat = SimpleNamespace(start=AsyncMock(), stop=Mock())
    channels = SimpleNamespace(start_all=AsyncMock(), stop_all=AsyncMock())
    from nanobot.cli import commands
    real_start = commands._start_dashboard_listener
    seen = []

    async def capture(*args):
        runner = await real_start(*args)
        seen.append(runner)
        return runner

    monkeypatch.setattr(commands, "_start_dashboard_listener", capture)
    await _run_gateway_services(agent, cron, heartbeat, channels, config)
    assert active.cancelled()
    assert seen[0] is not None and not seen[0].sites
    agent.close_mcp.assert_awaited_once()
    channels.stop_all.assert_awaited_once()
