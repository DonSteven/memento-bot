import json
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.memory_db import MemorySnapshot
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.tests.memory_test_utils import TestMemoryService as MemoryService

MESSAGES = [{"role": "user", "content": "Remember Nanobot"}]
FACT = {"main_class": "projects", "sub_class": "active", "text": "Nanobot project"}
VALID = {"history_entry": "saved", "canonical_memories": [FACT]}


def _response(arguments):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="save", name="save_memory_structured", arguments=arguments)],
    )


def _service(workspace, arguments):
    provider = AsyncMock()
    provider.chat_with_retry.return_value = _response(arguments)
    return MemoryService(workspace, provider, "test-model")


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [VALID, json.dumps(VALID), [VALID]])
async def test_supported_tool_argument_formats_commit_snapshot(tmp_path, arguments):
    service = _service(tmp_path, arguments)
    result = await service.consolidate(MESSAGES, session_key="test")
    assert result.database_committed and result.view_exported
    assert service.database.read_snapshot().memories[0].text == FACT["text"]
    assert service.database.history_file.read_text().strip() == "saved"
    kwargs = service.pipeline.provider.chat_with_retry.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert not {"temperature", "max_tokens", "reasoning_effort"} & kwargs.keys()


@pytest.mark.asyncio
async def test_structured_text_values_are_serialized(tmp_path):
    service = _service(
        tmp_path,
        {
            "history_entry": {"summary": "saved"},
            "canonical_memories": [{**FACT, "text": {"name": "Nanobot"}}],
        },
    )
    result = await service.consolidate(MESSAGES, session_key="test")
    assert result.database_committed
    assert json.loads(service.database.history_file.read_text()) == {"summary": "saved"}
    assert json.loads(service.database.read_snapshot().memories[0].text) == {"name": "Nanobot"}


INVALID = [
    None,
    "{bad json",
    [],
    ["invalid"],
    {},
    {"canonical_memories": []},
    {"history_entry": "saved"},
    {**VALID, "history_entry": None},
    {**VALID, "history_entry": "   "},
    {**VALID, "canonical_memories": None},
    {**VALID, "canonical_memories": {}},
    *[
        {**VALID, "canonical_memories": [FACT, invalid]}
        for invalid in (
            "invalid",
            {**FACT, "main_class": "unknown"},
            {**FACT, "sub_class": ""},
            {**FACT, "sub_class": None},
            {**FACT, "text": " "},
            {**FACT, "text": None},
            {"main_class": "projects", "text": "missing subclass"},
        )
    ],
]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", INVALID)
async def test_invalid_extraction_does_not_change_snapshot_event_index_or_views(
    tmp_path, arguments
):
    service = _service(tmp_path, VALID)
    await service.consolidate(MESSAGES, session_key="test")
    db = service.database
    with db.connect() as conn:
        before = list(conn.iterdump())
    views = (db.memory_file.read_bytes(), db.history_file.read_bytes())
    service.pipeline.provider.chat_with_retry.return_value = _response(arguments)

    result = await service.consolidate(MESSAGES, session_key="test")

    assert not result.database_committed and result.error
    with db.connect() as conn:
        assert list(conn.iterdump()) == before
    assert (db.memory_file.read_bytes(), db.history_file.read_bytes()) == views


@pytest.mark.asyncio
async def test_snapshot_replacement_and_empty_snapshot_remove_old_fts_hits(tmp_path):
    service = _service(tmp_path, VALID)
    await service.consolidate(MESSAGES, session_key="test")
    provider = service.pipeline.provider
    provider.chat_with_retry.return_value = _response(
        {
            "history_entry": "replaced",
            "canonical_memories": [{**FACT, "text": "Orion project"}],
        }
    )
    result = await service.consolidate(MESSAGES, session_key="test")
    assert result.database_committed
    prompt = provider.chat_with_retry.call_args.kwargs["messages"][1]["content"]
    assert "Nanobot project" in prompt
    assert service.database.query_dynamic_memories("Nanobot") == ()
    assert len(service.database.query_dynamic_memories("Orion")) == 1
    provider.chat_with_retry.return_value = _response(
        {"history_entry": "cleared", "canonical_memories": []}
    )
    result = await service.consolidate(MESSAGES, session_key="test")
    assert result.database_committed
    assert service.database.read_snapshot() == MemorySnapshot(3)
    assert service.database.query_dynamic_memories("Orion") == ()
    assert "Orion project" not in service.database.memory_file.read_text()
    assert len(service.database.list_raw_events()) == 3


@pytest.mark.asyncio
async def test_empty_conversation_does_not_call_provider_or_write(tmp_path):
    service = _service(tmp_path, VALID)
    result = await service.consolidate([], session_key="test")
    assert result.database_committed
    service.pipeline.provider.chat_with_retry.assert_not_awaited()
    assert service.database.read_snapshot() == MemorySnapshot(0)
    assert service.database.list_raw_events() == []
    assert service.database.memory_file.exists()


@pytest.mark.asyncio
async def test_tool_choice_retry_without_tool_call_does_not_commit(tmp_path):
    service = _service(tmp_path, VALID)
    service.pipeline.provider.chat_with_retry.side_effect = [
        LLMResponse(content="tool_choice unsupported", finish_reason="error"),
        LLMResponse(content="no tool call"),
    ]
    result = await service.consolidate(MESSAGES, session_key="test")
    assert not result.database_committed
    assert service.database.read_snapshot() == MemorySnapshot(0)
    assert service.database.list_raw_events() == []
    assert service.pipeline.provider.chat_with_retry.await_count == 2
