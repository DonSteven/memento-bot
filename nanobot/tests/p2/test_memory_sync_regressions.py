from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent import memory_sync
from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory_db import MemoryRecord, MemorySnapshot
from nanobot.agent.memory_service import MemoryService
from nanobot.agent.memory_sync import parse_memory_markdown, render_memory_markdown
from nanobot.api.server import handle_chat_completions
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _response(sub_class="style", text="Use Python"):
    return LLMResponse(content=None, tool_calls=[ToolCallRequest(
        id="save", name="save_memory_structured", arguments={
            "history_entry": "Saved development preferences",
            "canonical_memories": [
                {"main_class": "preferences", "sub_class": sub_class, "text": text},
            ],
        },
    )])


@pytest.mark.asyncio
@pytest.mark.parametrize(("sub_class", "text"), [
    ("style", "Use Python.\nPrefer pytest."),
    ("style", "Use Python.\n- testing: Prefer pytest"),
    ("project: alpha", "Uses Python"),
    ("project\nalpha", "Uses Python"),
    ("style", "Use Python.\rPrefer pytest."),
    ("style", "Use Python.\u2028Prefer pytest."),
])
async def test_unrepresentable_extraction_preserves_database_and_views(tmp_path, sub_class, text):
    provider = AsyncMock()
    provider.chat_with_retry.return_value = _response()
    service = MemoryService(tmp_path, provider, "test")
    messages = [{"role": "user", "content": "Remember my development preferences"}]
    await service.consolidate(messages, session_key="test")
    db = service.database
    with db.connect() as conn:
        before = list(conn.iterdump())
    views = (db.memory_file.read_bytes(), db.history_file.read_bytes())

    provider.chat_with_retry.return_value = _response(sub_class, text)
    result = await service.consolidate(messages, session_key="test")

    assert not result.database_committed and not result.view_exported
    assert "Invalid canonical memory" in result.error
    with db.connect() as conn:
        assert list(conn.iterdump()) == before
    assert (db.memory_file.read_bytes(), db.history_file.read_bytes()) == views

    # A harmless editor save must still preserve the previously accepted snapshot.
    snapshot = db.read_snapshot()
    db.memory_file.write_text(db.memory_file.read_text() + "\n", encoding="utf-8")
    await service.prepare_context("hello")
    assert db.read_snapshot() == snapshot
    assert (db.memory_file.read_bytes(), db.history_file.read_bytes()) == views


def test_direct_records_cannot_bypass_markdown_validation(tmp_path):
    service = MemoryService(tmp_path, AsyncMock(), "test")
    invalid = MemoryRecord("custom-id", "preferences", "project: alpha", "Uses Python")
    with pytest.raises(ValueError, match="delimiter"):
        render_memory_markdown((invalid,))
    with pytest.raises(ValueError, match="delimiter"):
        service.database.commit_snapshot(
            MemorySnapshot(0, (invalid,)), expected_revision=0, event_id="invalid",
            ts="2026-09-05", session_key="test", history_text="invalid",
        )
    assert service.database.read_snapshot() == MemorySnapshot(0)
    assert service.database.list_raw_events() == []


def test_single_line_punctuation_and_literal_escapes_round_trip():
    record = MemoryRecord.create("preferences", "project:alpha", r"Use URL: https://example.com; \n is literal")
    assert parse_memory_markdown(render_memory_markdown((record,))) == (record,)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["gateway", "http"])
@pytest.mark.parametrize("failure", ["invalid_markdown", "pending_conflict"])
async def test_sync_errors_reach_users_without_model_calls(tmp_path, entrypoint, failure):
    provider = AsyncMock()
    bus = MessageBus()
    agent = AgentLoop(bus, provider, tmp_path, model="test")
    await agent.memory_service.sync_markdown()
    db = agent.memory_service.database
    old_text = db.memory_file.read_text()
    if failure == "invalid_markdown":
        human_text = old_text.replace("## Projects", "## Invalid")
        expected = "MEMORY.md line 17: unknown memory category 'Invalid'"
    else:
        records = (MemoryRecord.create("preferences", "style", "Database target"),)
        db.commit_snapshot(
            MemorySnapshot(0, records), expected_revision=0, event_id="pending",
            ts="2026-09-05", session_key="test", history_text="pending",
            publish_expected_text=old_text, publish_target_text=render_memory_markdown(records),
            stage_publish=True,
        )
        human_text = old_text + "\n"
        expected = "the file was preserved and the database target remains pending"
    db.memory_file.write_text(human_text, encoding="utf-8")
    snapshot = db.read_snapshot()
    state = agent.memory_service.synchronizer.read_state()

    if entrypoint == "gateway":
        await agent._dispatch(InboundMessage(
            channel="cli", sender_id="test", chat_id="test", content="hello",
        ))
        response = bus.outbound.get_nowait()
        assert expected in response.content
        assert bus.outbound.empty()
    else:
        request = MagicMock()
        request.json = AsyncMock(return_value={"messages": [{"role": "user", "content": "hello"}]})
        request.app = {"agent_loop": agent, "session_locks": {}}
        response = await handle_chat_completions(request)
        assert response.status == 409
        error = json.loads(response.body)["error"]
        assert error["type"] == "memory_sync_error"
        assert expected in error["message"]

    provider.chat_with_retry.assert_not_awaited()
    assert db.read_snapshot() == snapshot
    assert db.memory_file.read_text() == human_text
    assert agent.memory_service.synchronizer.read_state() == state


@pytest.mark.asyncio
async def test_unchanged_queries_skip_history_io_and_rebuild_a_missing_view(tmp_path, monkeypatch):
    service = MemoryService(tmp_path, AsyncMock(), "test")
    await service.archive_raw([{"role": "user", "content": "Saved conversation"}], session_key="test")
    db = service.database
    original = db.history_file.read_bytes()
    stat = db.history_file.stat()
    snapshot = db.read_snapshot()
    with monkeypatch.context() as patch:
        patch.setattr(db, "render_history_view", MagicMock(side_effect=AssertionError("history read")))
        for _ in range(3):
            await service.prepare_context("hello")
    assert db.history_file.read_bytes() == original
    assert db.history_file.stat().st_mtime_ns == stat.st_mtime_ns
    assert db.history_file.stat().st_ino == stat.st_ino
    db.history_file.unlink()
    await service.prepare_context("hello")
    assert db.history_file.read_bytes() == original
    assert db.read_snapshot() == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_history_export_failure_remains_pending_and_recovers(tmp_path, monkeypatch, restart):
    provider = AsyncMock()
    provider.chat_with_retry.return_value = _response()
    service = MemoryService(tmp_path, provider, "test")
    await service.sync_markdown()
    db = service.database
    old_history = db.history_file.read_bytes()
    write = memory_sync._atomic_write

    def fail_history(path, text):
        if path == db.history_file:
            raise OSError("history disk full")
        return write(path, text)

    with monkeypatch.context() as patch:
        patch.setattr(memory_sync, "_atomic_write", fail_history)
        result = await service.consolidate(
            [{"role": "user", "content": "Remember Python"}], session_key="test",
        )
    assert result.database_committed and not result.view_exported
    assert result.error == "history disk full"
    assert service.synchronizer.read_state().pending_revision == 1
    assert db.history_file.read_bytes() == old_history
    snapshot = db.read_snapshot()
    assert db.memory_file.read_text() == render_memory_markdown(snapshot.memories)

    if restart:
        service = MemoryService(tmp_path, provider, "test")
    await service.prepare_context("hello")
    assert db.read_snapshot() == snapshot
    assert db.history_file.read_text().strip() == "Saved development preferences"
    assert service.synchronizer.read_state().pending_revision is None
    assert service.synchronizer.read_state().published_revision == 1
    assert len(db.list_raw_events()) == 1
    provider.chat_with_retry.assert_awaited_once()
