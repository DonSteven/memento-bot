from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.agent.memory_db import MemoryRecord, MemorySnapshot
from nanobot.agent.memory_sync import render_memory_markdown
from nanobot.api.server import create_app
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.nanobot import Nanobot
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.tests.knowledge.test_knowledge_online import (
    setup_online as online_fixture,  # noqa: F401
)
from nanobot.tests.memory_test_utils import TestAgentLoop
from nanobot.tests.p6.test_context_budget import provider


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["cli", "sdk", "http", "system"])
async def test_entry_sends_same_prepared_memory_once(tmp_path, entry):
    p = provider()
    p.chat_stream_with_retry = p.chat_with_retry
    loop = TestAgentLoop(MessageBus(), p, tmp_path)
    core = MemoryRecord.create("constraints", "rule", "Do not delete originals")
    dynamic = MemoryRecord.create("projects", "active", "Nanobot semantic memory")
    loop.memory_service.pipeline.extract_snapshot = AsyncMock(return_value=(MemorySnapshot(0, (core, dynamic)), "Remembered"))
    saved = await loop.memory_service.consolidate([{"role": "user", "content": "remember"}], session_key="seed")
    assert saved.database_committed and saved.view_exported
    if entry == "cli":
        result = await loop.process_direct("Nanobot", session_key="cli:test")
        assert result.content == "ok"
    elif entry == "sdk":
        result = await Nanobot(loop).run("Nanobot")
        assert result.content == "ok" and result.stop_reason == "completed"
    elif entry == "system":
        result = await loop._process_message(InboundMessage(channel="system", sender_id="subagent",
            chat_id="cli:test", content="Nanobot"))
        assert result.content == "ok"
    else:
        async with TestClient(TestServer(create_app(loop, model_name="test"))) as client:
            result = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Nanobot"}]})
            assert result.status == 200
            assert (await result.json())["choices"][0]["message"]["content"] == "ok"
    prompt = p.chat_with_retry.call_args.kwargs["messages"][0]["content"]
    assert prompt.count(core.text) == prompt.count(dynamic.text) == 1
    assert loop.memory_service.embedder.query_calls == ["Nanobot"]
    await loop.close_mcp()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["sdk", "http"])
async def test_context_limit_is_exposed_without_model_call(tmp_path, entry):
    p = provider()
    loop = TestAgentLoop(MessageBus(), p, tmp_path, context_window_tokens=100)
    if entry == "sdk":
        result = await Nanobot(loop).run("hello")
        assert result.stop_reason == "context_limit" and "limit" in result.content
    else:
        async with TestClient(TestServer(create_app(loop, model_name="test"))) as client:
            result = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})
            assert result.status == 400
            assert (await result.json())["error"]["type"] == "context_limit"
    p.chat_with_retry.assert_not_awaited()
    await loop.close_mcp()


@pytest.mark.asyncio
async def test_archive_edit_online_and_restart_through_agent(online_fixture, tmp_path):  # noqa: F811
    _, knowledge, p, search, fetch = online_fixture
    p.generation = GenerationSettings(max_tokens=100)
    captured = []
    assessment_chat = p.chat

    async def chat(**kwargs):
        if len(kwargs["tools"]) == 1 and kwargs["tools"][0]["function"]["name"] == "report_evidence_assessment":
            return await assessment_chat(**kwargs)
        captured.append(deepcopy(kwargs["messages"]))
        if kwargs["messages"][-1]["role"] == "tool":
            return LLMResponse(content="Linux answer")
        return LLMResponse(content=None, tool_calls=[ToolCallRequest(id="kb", name="kb_search", arguments={"query": "Linux release"})])

    p.chat_with_retry = AsyncMock(side_effect=chat)
    def build_loop():
        with patch("nanobot.agent.loop.WebKnowledgeService", return_value=knowledge):
            loop = TestAgentLoop(MessageBus(), p, tmp_path, knowledge_config=knowledge.config)
        loop.knowledge_retriever.search.search = search
        loop.knowledge_retriever.fetch.execute = fetch
        return loop

    loop = build_loop()
    old_core = MemoryRecord.create("preferences", "language", "Old preference")
    old_dynamic = MemoryRecord.create("projects", "active", "Old project")
    loop.memory_service.pipeline.extract_snapshot = AsyncMock(return_value=(MemorySnapshot(0, (old_core, old_dynamic)), "Archive summary"))
    session = loop.sessions.get_or_create("cli:archive")
    session.add_message("user", "Remember these preferences and project")
    loop.sessions.save(session)
    await loop.process_direct("/new", session_key="cli:archive")
    assert not session.messages
    core = MemoryRecord.create("preferences", "language", "Answer in Chinese")
    dynamic = MemoryRecord.create("projects", "active", "Nanobot semantic memory")
    loop.memory_service.database.memory_file.write_text(render_memory_markdown((core, dynamic)), encoding="utf-8")
    await loop.process_direct("Nanobot", session_key="cli:question")
    assert len(knowledge.db.list_pages()) == 1
    assert search.await_count == fetch.await_count == 1
    for messages in captured:
        assert core.text in messages[0]["content"] and dynamic.text in messages[0]["content"]
        assert old_core.text not in messages[0]["content"] and old_dynamic.text not in messages[0]["content"]
    assert any(m["role"] == "tool" and "Linux 6.9" in m["content"] for m in captured[-1])
    # Commit another core edit, deliberately leave the old file to simulate interrupted publishing.
    db = loop.memory_service.database
    snapshot = db.read_snapshot()
    new_core = MemoryRecord.create("preferences", "language", "Use concise Chinese")
    db.commit_snapshot(MemorySnapshot(snapshot.revision, (new_core, dynamic)),
        expected_revision=snapshot.revision, event_id="interrupted", ts="now", session_key="cli:archive",
        history_text="Committed before restart", stage_publish=True,
        publish_expected_text=db.memory_file.read_text(), publish_target_text=render_memory_markdown((new_core, dynamic)))
    await loop.close_mcp()
    restarted = build_loop()
    await restarted.process_direct("Nanobot", session_key="cli:restart")
    assert new_core.text in captured[-1][0]["content"]
    assert core.text not in captured[-1][0]["content"]
    assert new_core.text in db.memory_file.read_text()
    assert search.await_count == fetch.await_count == 1  # persisted local evidence suffices
    assert restarted.memory_service.synchronizer.read_state().pending_revision is None
    await restarted.close_mcp()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["direct", "bus"])
async def test_streaming_tool_overflow_delivers_error_and_final_end(tmp_path, entry):
    p = provider()

    async def stream(**kwargs):
        await kwargs["on_content_delta"]("Checking the tool.")
        return LLMResponse(content="Checking the tool.", tool_calls=[
            ToolCallRequest(id="large", name="large", arguments={})])

    p.chat_stream_with_retry = AsyncMock(side_effect=stream)
    loop = TestAgentLoop(MessageBus(), p, tmp_path, context_window_tokens=6000)
    loop.tools.execute = AsyncMock(return_value="large response " * 20000)
    events = []
    if entry == "direct":
        async def delta(content):
            events.append(("delta", content))

        async def end(*, resuming):
            events.append(("end", resuming))

        response = await loop.process_direct("hello", on_stream=delta, on_stream_end=end)
    else:
        await loop._dispatch(InboundMessage(channel="cli", sender_id="user", chat_id="test",
            content="hello", metadata={"_wants_stream": True}))
        while True:
            response = await loop.bus.consume_outbound()
            if response.metadata.get("_stream_delta"):
                events.append(("delta", response.content))
            elif response.metadata.get("_stream_end"):
                events.append(("end", response.metadata["_resuming"]))
            elif response.metadata.get("stop_reason"):
                break
    assert response.metadata["stop_reason"] == "context_limit"
    assert response.metadata["_streamed"]
    assert events == [("delta", "Checking the tool."), ("end", True),
                      ("delta", response.content), ("end", False)]
    assert "Context limit exceeded" in response.content
    assert p.chat_stream_with_retry.await_count == 1
    await loop.close_mcp()
