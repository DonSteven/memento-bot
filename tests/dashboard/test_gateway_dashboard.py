import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.cli.commands import _run_gateway_services, _start_dashboard_listener
from nanobot.config.schema import Config
from nanobot.observability.store import ObservabilityStore


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
