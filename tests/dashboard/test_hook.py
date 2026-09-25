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


@pytest.mark.parametrize("value,expected", [
    ({"accessToken": "DEMO_SECRET_DO_NOT_PERSIST", "prompt_tokens": 12}, '"prompt_tokens": 12'),
    ({"items": [{"X-API-Key": "DEMO_SECRET_DO_NOT_PERSIST"}], "id": "demo-1"}, '"id": "demo-1"'),
    ('{"refresh_token":"DEMO_SECRET_DO_NOT_PERSIST","completion_tokens":3}', '"completion_tokens": 3'),
    ('data:image/png;base64,DEMO_SECRET_DO_NOT_PERSIST', "[BINARY DATA]"),
    ({"image": "data:image/png;base64,DEMO_SECRET_DO_NOT_PERSIST"}, "[BINARY DATA]"),
    ("image=data:image/png;base64,DEMO_SECRET_DO_NOT_PERSIST done", "done"),
    ("Authorization: Bearer DEMO_SECRET_DO_NOT_PERSIST\nordinary prose", "ordinary prose"),
    ("Authorization: Basic DEMO_SECRET_DO_NOT_PERSIST", "[REDACTED]"),
    ("Cookie: session=DEMO_SECRET_DO_NOT_PERSIST", "[REDACTED]"),
    ("Set-Cookie: session=DEMO_SECRET_DO_NOT_PERSIST; HttpOnly", "[REDACTED]"),
    ("client_secret='DEMO_SECRET_DO_NOT_PERSIST' count=2", "count=2"),
    ('x_api_key: "DEMO_SECRET_DO_NOT_PERSIST"', "[REDACTED]"),
    ('{"broken": "token=DEMO_SECRET_DO_NOT_PERSIST', "[REDACTED]"),
])
def test_preview_supported_formats(value, expected):
    result = preview(value)
    assert "DEMO_SECRET_DO_NOT_PERSIST" not in result["text"]
    assert expected in result["text"]
    assert result["truncated"] is False


def test_preview_preserves_input_and_nonsensitive_text():
    value = {"nested": [{"clientSecret": "DEMO_SECRET_DO_NOT_PERSIST"}],
             "prompt_tokens": 24, "run_id": "demo-run", "text": "ordinary prose"}
    output = preview(value)["text"]
    assert value["nested"][0]["clientSecret"] == "DEMO_SECRET_DO_NOT_PERSIST"
    assert all(item in output for item in ("prompt_tokens", "24", "demo-run", "ordinary prose"))
    assert preview(output)["text"] == output


def test_preview_sanitizes_before_utf8_truncation():
    value = "é" * 4 + " token=DEMO_SECRET_DO_NOT_PERSIST"
    result = preview(value, 18)
    assert result["truncated"] is True
    assert len(result["text"].encode("utf-8")) <= 18
    assert "DEMO_SECRET" not in result["text"]
    assert result["text"].startswith("é" * 4)


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
