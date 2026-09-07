from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory_db import MemoryRecord, MemorySnapshot
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.tests.memory_test_utils import TestAgentLoop as AgentLoop


def _loop(workspace):
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (50, "test")
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok"))
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="ok"))
    loop = AgentLoop(MessageBus(), provider, workspace, context_window_tokens=200)
    loop.memory_consolidator._SAFETY_BUFFER = 0
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.memory_service.database.commit_snapshot(
        MemorySnapshot(0, (MemoryRecord.create("projects", "active", "Nanobot old fact"),)),
        expected_revision=0,
        event_id="seed",
        ts="2026-09-05",
        session_key="cli:test",
        history_text="seed",
        dynamic_embeddings={
            MemoryRecord.create("projects", "active", "Nanobot old fact").memory_id: [1.0, 0.0, 0.0]
        },
    )
    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second turn"},
    ]
    return loop, session


@pytest.mark.asyncio
async def test_estimation_uses_prepared_dynamic_memory_without_queries(tmp_path, monkeypatch):
    loop, session = _loop(tmp_path)
    prepared = await loop.memory_service.prepare_context("Nanobot")
    assert len(prepared.retrieved_items) == 1
    monkeypatch.setattr(
        loop.memory_service.database, "connect", MagicMock(side_effect=AssertionError("DB read"))
    )
    monkeypatch.setattr(
        loop.memory_service, "prepare_context", AsyncMock(side_effect=AssertionError("retrieval"))
    )

    result = loop.memory_consolidator.estimate_session_prompt_tokens(
        session,
        prepared,
        current_message="Nanobot",
    )

    assert result == (50, "test")
    prompt = loop.provider.estimate_prompt_tokens.call_args.args[0]
    assert "Nanobot old fact" in prompt[0]["content"]
    loop.memory_service.prepare_context.assert_not_awaited()
    loop.provider.chat_with_retry.assert_not_awaited()
    loop.provider.chat_stream_with_retry.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("system_message", [False, True])
@pytest.mark.parametrize("archive", [False, True])
async def test_loop_reuses_prepared_memory_and_refreshes_after_commit(
    tmp_path,
    monkeypatch,
    system_message,
    archive,
):
    loop, session = _loop(tmp_path)
    service = loop.memory_service
    prepare = AsyncMock(wraps=service.prepare_context)
    monkeypatch.setattr(service, "prepare_context", prepare)
    contexts = []
    build_messages = loop.context.build_messages

    def capture(*args, **kwargs):
        contexts.append(kwargs["memory_context"])
        return build_messages(*args, **kwargs)

    monkeypatch.setattr(loop.context, "build_messages", capture)
    loop.memory_consolidator._build_messages = capture
    new_record = MemoryRecord.create("projects", "active", "Nanobot new fact")
    extraction = AsyncMock(return_value=(MemorySnapshot(1, (new_record,)), "updated"))
    monkeypatch.setattr(service.pipeline, "extract_snapshot", extraction)
    if archive:
        loop.provider.estimate_prompt_tokens.side_effect = [
            (50, "test"),  # P6 necessary-content preflight
            (300, "test"),
            (50, "test"),
            (50, "test"),
            (50, "test"),
            (50, "test"),
        ]
        monkeypatch.setattr("nanobot.agent.memory.estimate_message_tokens", lambda _: 500)
    message = InboundMessage(
        channel="system" if system_message else "cli",
        sender_id="subagent" if system_message else "user",
        chat_id="cli:test" if system_message else "test",
        content="Nanobot",
    )

    await loop._process_message(message)
    await loop.close_mcp()

    assert prepare.await_count == (2 if archive else 1)
    assert all(call.args == ("Nanobot",) for call in prepare.await_args_list)
    # Each estimate, the actual prompt, and the background check share the prepared object.
    assert all(context is contexts[-1] for context in contexts[2 if archive else 0 :])
    assert contexts[-1].retrieved_items[0].text == (
        "Nanobot new fact" if archive else "Nanobot old fact"
    )
    assert extraction.await_count == int(archive)
    assert session.last_consolidated == (2 if archive else 0)
    if archive:
        assert contexts[0].retrieved_items[0].text == "Nanobot old fact"
        assert service.database.read_snapshot().revision == 2
