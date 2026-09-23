import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.memory_db import MemoryContext
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.observability.store import ObservabilityStore
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.tests.memory_test_utils import TestAgentLoop as AgentLoop


def make_loop(tmp_path, responses):
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (10, "test")
    provider.chat_with_retry = AsyncMock(side_effect=responses)
    loop = AgentLoop(MessageBus(), provider, tmp_path, context_window_tokens=100000)
    loop.memory_service.prepare_context = AsyncMock(return_value=MemoryContext())
    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock(
        side_effect=lambda session, memory_context, **kwargs: memory_context)
    loop.tools.execute = AsyncMock(return_value="x" * 5000)
    return loop


@pytest.mark.asyncio
async def test_two_round_real_turn_and_usage(tmp_path):
    loop = make_loop(tmp_path, [
        LLMResponse(content="working", tool_calls=[
            ToolCallRequest(id="c1", name="list_dir", arguments={"path": "."})],
            usage={"prompt_tokens": 10, "completion_tokens": 2}),
        LLMResponse(content="done", usage={"prompt_tokens": 20, "completion_tokens": 3}),
    ])
    response = await loop.process_direct("hello", session_key="cli:test")
    assert response.content == "done"
    store = ObservabilityStore(tmp_path / "observability" / "dashboard.db")
    run = store.list_runs()["items"][0]
    assert (run["status"], run["stop_reason"]) == ("completed", "completed")
    assert (run["prompt_tokens"], run["completion_tokens"]) == (30, 5)
    events = store.get_run(run["run_id"])["events"]
    assert [event["kind"] for event in events] == ["memory", "model", "tools", "model"]
    assert events[2]["data"]["calls"][0]["result_preview"]["truncated"]
    await loop.close_mcp()


@pytest.mark.asyncio
async def test_memory_error_and_cancelled_are_terminal(tmp_path):
    loop = make_loop(tmp_path, [LLMResponse(content="unused")])
    loop.memory_service.prepare_context = AsyncMock(side_effect=RuntimeError("memory failed"))
    with pytest.raises(RuntimeError, match="memory failed"):
        await loop.process_direct("hello", session_key="cli:error")
    store = ObservabilityStore(tmp_path / "observability" / "dashboard.db")
    detail = store.get_run(store.list_runs()["items"][0]["run_id"])
    assert detail["run"]["status"] == "error"
    assert detail["events"][0]["kind"] == "memory"
    loop.memory_service.prepare_context = AsyncMock(return_value=MemoryContext())
    gate = asyncio.Event()

    async def wait_provider(**kwargs):
        await gate.wait()

    loop.provider.chat_with_retry = wait_provider
    task = asyncio.create_task(loop.process_direct("hello", session_key="cli:cancel"))
    while len(store.list_runs()["items"]) < 2:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.list_runs()["items"][0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_locked_store_does_not_block_event_loop(tmp_path):
    store = ObservabilityStore(tmp_path / "observability" / "dashboard.db")
    store.initialize()
    loop = make_loop(tmp_path, [LLMResponse(content="done")])
    assert await loop._initialize_observability()
    db = sqlite3.connect(store.path)
    db.execute("BEGIN IMMEDIATE")
    turn = asyncio.create_task(loop.process_direct("hello", session_key="cli:lock"))
    await asyncio.sleep(0.05)
    tick = asyncio.Event()
    asyncio.get_running_loop().call_soon(tick.set)
    await asyncio.wait_for(tick.wait(), timeout=0.2)
    db.rollback()
    db.close()
    assert (await turn).content == "done"
    await loop.close_mcp()


@pytest.mark.asyncio
async def test_stop_reasons_and_no_fake_model_on_context_limit(tmp_path):
    loop = make_loop(tmp_path, [LLMResponse(content="provider failed", finish_reason="error")])
    await loop.process_direct("failure", session_key="cli:error")
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working", tool_calls=[ToolCallRequest(id="c", name="list_dir", arguments={})]))
    loop.max_iterations = 1
    await loop.process_direct("iteration", session_key="cli:iteration")
    from nanobot.agent.context_budget import ContextBudgetError
    with patch("nanobot.agent.runner.fit_request", side_effect=ContextBudgetError("too large")):
        await loop.process_direct("budget", session_key="cli:budget")
    store = ObservabilityStore(tmp_path / "observability" / "dashboard.db")
    details = [store.get_run(run["run_id"]) for run in store.list_runs()["items"]]
    assert {item["run"]["stop_reason"]: item["run"]["status"] for item in details} == {
        "error": "error", "max_iterations": "stopped", "context_limit": "stopped"}
    budget = next(item for item in details if item["run"]["stop_reason"] == "context_limit")
    assert [event["kind"] for event in budget["events"]] == ["memory", "error"]
    await loop.close_mcp()


@pytest.mark.asyncio
async def test_concurrent_sessions_and_store_failure_isolation(tmp_path):
    loop = make_loop(tmp_path, [])

    async def respond(*, messages, **kwargs):
        await asyncio.sleep(0.01)
        if any("bad" in str(message.get("content")) for message in messages):
            raise RuntimeError("provider failed")
        return LLMResponse(content="good", usage={"prompt_tokens": 7,
                                                  "completion_tokens": 1})

    loop.provider.chat_with_retry = respond
    good, bad = await asyncio.gather(
        loop.process_direct("good", session_key="cli:good"),
        loop.process_direct("bad", session_key="cli:bad"),
        return_exceptions=True)
    assert good.content == "good"
    assert isinstance(bad, RuntimeError)
    store = ObservabilityStore(tmp_path / "observability" / "dashboard.db")
    by_session = {run["session_key"]: run for run in store.list_runs()["items"]}
    assert by_session["cli:good"]["status"] == "completed"
    assert by_session["cli:good"]["prompt_tokens"] == 7
    assert by_session["cli:bad"]["status"] == "error"
    assert by_session["cli:bad"]["prompt_tokens"] == 0
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="still works"))
    loop.observability_store.initialize = MagicMock(side_effect=sqlite3.OperationalError("disk"))
    loop._observability_initialized = False
    assert (await loop.process_direct("hello", session_key="cli:unobserved")).content == "still works"
    await loop.close_mcp()


@pytest.mark.asyncio
async def test_system_message_is_traced_but_command_is_not(tmp_path):
    loop = make_loop(tmp_path, [LLMResponse(content="system done")])
    await loop.process_direct("/status", session_key="cli:test")
    path = tmp_path / "observability" / "dashboard.db"
    assert not path.exists()
    message = InboundMessage(channel="system", sender_id="cron", chat_id="cli:test",
                             content="check")
    response = await loop._process_message(message)
    assert response.content == "system done"
    store = ObservabilityStore(path)
    assert store.list_runs()["items"][0]["session_key"] == "cli:test"
    await loop.close_mcp()
