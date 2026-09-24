import asyncio
import json
from dataclasses import asdict

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.agent.tools.cron import CronTool
from nanobot.api.dashboard import CRON_KEY, create_dashboard_app
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.observability.store import ObservabilityStore


@pytest.mark.asyncio
async def test_tasks_use_live_cron_service_and_persist(tmp_path):
    path = tmp_path / "cron" / "jobs.json"
    cron = CronService(path, on_job=lambda _: asyncio.sleep(0))
    recurring = cron.add_job("Daily summary", CronSchedule(kind="cron", expr="0 9 * * *", tz="UTC"), "summarize")
    one_shot = cron.add_job("Reminder", CronSchedule(kind="at", at_ms=4102444800000), "remind")
    await cron.run_job(recurring.id)
    cron.enable_job(one_shot.id, False)
    await cron.start()
    tool = CronTool(cron)
    app = create_dashboard_app(object(), cron, ObservabilityStore(tmp_path / "dashboard.db"))
    assert app[CRON_KEY] is cron
    assert tool._cron is cron
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/api/dashboard/tasks")
            assert response.status == 200
            items = (await response.json())["items"]
            assert {item["id"] for item in items} == {recurring.id, one_shot.id}
            assert next(item for item in items if item["id"] == recurring.id) == asdict(cron.get_job(recurring.id))
            assert next(item for item in items if item["id"] == one_shot.id)["enabled"] is False
            assert items[0]["state"]["run_history"][0]["status"] == "ok"

            response = await client.patch(f"/api/dashboard/tasks/{recurring.id}", json={"enabled": False})
            assert response.status == 200
            assert (await response.json())["state"]["next_run_at_ms"] is None
            assert cron.get_job(recurring.id).enabled is False
            assert recurring.id not in await tool.execute(action="list")

            response = await client.patch(f"/api/dashboard/tasks/{recurring.id}", json={"enabled": True})
            assert response.status == 200
            updated = await response.json()
            assert updated["enabled"] is True
            assert updated["state"]["next_run_at_ms"] is not None
            assert cron.get_job(recurring.id).enabled is True

            response = await client.delete(f"/api/dashboard/tasks/{one_shot.id}")
            assert response.status == 204
            assert await response.read() == b""
            assert cron.get_job(one_shot.id) is None
            assert {item["id"] for item in (await (await client.get("/api/dashboard/tasks")).json())["items"]} == {recurring.id}

        fresh = CronService(path)
        assert asdict(fresh.get_job(recurring.id)) == asdict(cron.get_job(recurring.id))
        assert fresh.get_job(one_shot.id) is None
        raw = json.loads(path.read_text())
        assert set(raw) == {"version", "jobs"}
        assert set(raw["jobs"][0]) == {"id", "name", "enabled", "schedule", "payload", "state", "createdAtMs", "updatedAtMs", "deleteAfterRun"}
        assert "runHistory" in raw["jobs"][0]["state"]
    finally:
        cron.stop()


@pytest.mark.asyncio
async def test_tasks_validate_mutations_and_missing_ids(tmp_path):
    cron = CronService(tmp_path / "jobs.json")
    job = cron.add_job("Keep", CronSchedule(kind="every", every_ms=60000), "work")
    app = create_dashboard_app(object(), cron, ObservabilityStore(tmp_path / "dashboard.db"))
    async with TestClient(TestServer(app)) as client:
        for body in ({"enabled": 1}, {"enabled": "false"}, {"enabled": None}, {},
                     {"enabled": False, "name": "changed"}, []):
            response = await client.patch(f"/api/dashboard/tasks/{job.id}", json=body)
            assert response.status == 400
        response = await client.patch(f"/api/dashboard/tasks/{job.id}", data="not json")
        assert response.status == 400
        assert cron.get_job(job.id).enabled is True
        assert (await client.patch("/api/dashboard/tasks/missing", json={"enabled": False})).status == 404
        assert (await client.delete("/api/dashboard/tasks/missing")).status == 404

        response = await client.patch(f"/api/dashboard/tasks/{job.id}", json={"enabled": False},
                                      headers={"Origin": "http://other.test"})
        assert response.status == 403
        assert cron.get_job(job.id).enabled is True

    async with TestClient(TestServer(create_dashboard_app(object(), None,
                                                    ObservabilityStore(tmp_path / "other.db")))) as client:
        assert (await client.get("/api/dashboard/tasks")).status == 503
        assert (await client.patch("/api/dashboard/tasks/x", json={"enabled": True})).status == 503
        assert (await client.delete("/api/dashboard/tasks/x")).status == 503
