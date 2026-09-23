import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.observability.hook import ObservabilityHook, preview
from nanobot.observability.store import ObservabilityStore
from nanobot.providers.base import LLMResponse, ToolCallRequest


def test_preview_redacts_and_truncates():
    result = preview({"password": "private", "nested": {"api_key": "secret"},
                      "text": "é" * 3000})
    assert "private" not in result["text"]
    assert "secret" not in result["text"]
    assert result["truncated"] is True
    assert len(result["text"].encode()) <= 4096


@pytest.mark.asyncio
async def test_hook_keeps_repeated_tool_calls_in_order(tmp_path):
    store = ObservabilityStore(tmp_path / "dashboard.db")
    store.initialize()
    store.start_run("r", "cli:x", "cli", "test")
    hook = ObservabilityHook(store, "r")
    calls = [ToolCallRequest(id=f"c{i}", name="list_dir", arguments={"token": "secret"})
             for i in range(2)]
    context = AgentHookContext(iteration=0, messages=[])
    await hook.before_iteration(context)
    context.response = LLMResponse(content="working", tool_calls=calls,
                                   usage={"prompt_tokens": 10, "completion_tokens": 2})
    context.usage = {"prompt_tokens": 10, "completion_tokens": 2}
    context.tool_calls = calls
    await hook.before_execute_tools(context)
    context.tool_results = ["x" * 5000, "ok"]
    context.tool_events = [{"status": "error"}, {"status": "ok"}]
    await hook.after_iteration(context)
    detail = store.get_run("r")
    assert [event["kind"] for event in detail["events"]] == ["model", "tools"]
    batch = detail["events"][1]["data"]["calls"]
    assert [call["call_id"] for call in batch] == ["c0", "c1"]
    assert batch[0]["result_preview"]["truncated"] is True
    assert "secret" not in batch[0]["arguments"]["text"]
    assert detail["run"]["prompt_tokens"] == 10
