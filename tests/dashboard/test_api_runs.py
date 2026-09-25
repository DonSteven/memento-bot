import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.agent.hook import AgentHookContext
from nanobot.api.dashboard import AGENT_KEY, CRON_KEY, create_dashboard_app
from nanobot.observability.hook import ObservabilityHook
from nanobot.observability.store import ObservabilityStore
from nanobot.providers.base import LLMResponse, ToolCallRequest


@pytest.mark.asyncio
async def test_runs_rest_reads_injected_store_and_cron(tmp_path):
    store = ObservabilityStore(tmp_path / "dashboard.db")
    store.initialize()
    store.start_run("r1", "cli:test", "cli", "test-model")
    store.add_event("r1", "memory", {"core_count": 1})
    store.finish_run("r1", "completed", "completed", 12)

    class Cron:
        def list_jobs(self, include_disabled=False):
            assert include_disabled is True
            return [type("Job", (), {"enabled": True})(),
                    type("Job", (), {"enabled": False})()]

    agent = object()
    cron = Cron()
    app = create_dashboard_app(agent, cron, store)
    assert app[AGENT_KEY] is agent
    assert app[CRON_KEY] is cron
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/api/dashboard/overview")
        assert response.status == 200
        overview = await response.json()
        assert overview["runs"] == 1
        assert overview["enabled_tasks"] == 1
        response = await client.get("/api/dashboard/runs?limit=1")
        assert response.status == 200
        assert (await response.json())["items"][0]["run_id"] == "r1"
        response = await client.get("/api/dashboard/runs/r1")
        assert response.status == 200
        assert (await response.json())["events"][0]["data"]["core_count"] == 1
        assert (await client.get("/api/dashboard/runs/missing")).status == 404
        assert (await client.get("/api/dashboard/runs?limit=101")).status == 400
        assert (await client.get("/api/dashboard/runs?cursor=bad")).status == 400
        assert (await client.get("/api/dashboard/runs", headers={"Host": "evil.test"})).status == 400


@pytest.mark.asyncio
async def test_unavailable_store_is_503(tmp_path):
    store = ObservabilityStore(tmp_path / "missing" / "dashboard.db")
    app = create_dashboard_app(object(), object(), store)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/api/dashboard/runs")
        assert response.status == 503
        assert (await response.json())["error"]["code"] == "store_unavailable"


@pytest.mark.asyncio
async def test_new_observations_are_sanitized_in_store_and_rest(tmp_path):
    store = ObservabilityStore(tmp_path / "dashboard.db")
    store.initialize()
    store.start_run("demo", "cli:demo", "cli", "fake-model")
    hook = ObservabilityHook(store, "demo")
    call = ToolCallRequest(id="call-1", name="demo_tool", arguments={
        "clientSecret": "DEMO_SECRET_DO_NOT_PERSIST", "count": 2})
    context = AgentHookContext(iteration=0, messages=[])
    await hook.before_iteration(context)
    context.response = LLMResponse(content="Authorization: Bearer DEMO_SECRET_DO_NOT_PERSIST",
                                   tool_calls=[call], usage={"prompt_tokens": 11,
                                                             "completion_tokens": 3})
    context.usage = {"prompt_tokens": 11, "completion_tokens": 3}
    context.tool_calls = [call]
    await hook.before_execute_tools(context)
    context.tool_results = ["data:image/png;base64,DEMO_SECRET_DO_NOT_PERSIST"]
    context.tool_events = [{"status": "ok"}]
    await hook.after_iteration(context)
    store.finish_run("demo", "completed", "completed", 10)

    class Cron:
        def list_jobs(self, include_disabled=False):
            return []

    raw = store.get_run("demo")
    app = create_dashboard_app(object(), Cron(), store)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/api/dashboard/runs/demo")
        assert response.status == 200
        api = await response.json()
    for detail in (raw, api):
        assert "DEMO_SECRET_DO_NOT_PERSIST" not in str(detail)
        assert "[REDACTED]" in str(detail)
        assert "[BINARY DATA]" in str(detail)
        assert detail["run"]["prompt_tokens"] == 11
        assert [event["kind"] for event in detail["events"]] == ["model", "tools"]
        assert detail["events"][1]["data"]["calls"][0]["arguments"]["text"].find('"count": 2') > 0
    assert context.response.content == "Authorization: Bearer DEMO_SECRET_DO_NOT_PERSIST"
    assert context.tool_results == ["data:image/png;base64,DEMO_SECRET_DO_NOT_PERSIST"]
