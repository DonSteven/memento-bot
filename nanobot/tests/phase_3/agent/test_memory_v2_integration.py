# Phase 3 test content:
# - verifies v2 mode persists structured memory through memory.db and re-renders MEMORY.md/HISTORY.md
# - verifies canonical memories and raw events are written into the database
# - verifies repeated DB write failures degrade to raw archive instead of reviving the legacy chain
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_3/agent/test_memory_v2_integration.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "memory" / f"{name}.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _provider_response(arguments: dict) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_v2_1",
                name="save_memory_structured",
                arguments=arguments,
            )
        ],
    )


@pytest.mark.asyncio
async def test_v2_mode_writes_db_backed_views(tmp_path: Path) -> None:
    fixture = _load_fixture("v2_payload")
    store = MemoryStore(tmp_path, mode="v2")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=_provider_response(fixture["tool_arguments"]))

    result = await store.consolidate(fixture["messages"], provider, "test-model")

    assert result is True
    assert store.v2_db is not None
    assert store.v2_db.db_path.exists()
    assert store.memory_file.read_text(encoding="utf-8") == fixture["expected_memory_markdown"]
    assert store.history_file.read_text(encoding="utf-8") == fixture["expected_history_markdown"]

    canonical = store.v2_db.list_canonical_memories()
    raw_events = store.v2_db.list_raw_events()

    assert [item["main_class"] for item in canonical] == fixture["expected"]["main_classes"]
    assert [item["sub_class"] for item in canonical] == fixture["expected"]["sub_classes"]
    assert [item["status"] for item in canonical] == fixture["expected"]["statuses"]
    assert [item["text"] for item in canonical] == fixture["expected"]["texts"]
    assert len(raw_events) == 1
    assert raw_events[0]["history_text"] == fixture["tool_arguments"]["history_entry"]
    assert raw_events[0]["candidate_type"] == "v2_snapshot"


@pytest.mark.asyncio
async def test_v2_mode_raw_archives_after_repeated_db_write_failures(tmp_path: Path) -> None:
    fixture = _load_fixture("v2_payload")
    store = MemoryStore(tmp_path, mode="v2")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=_provider_response(fixture["tool_arguments"]))
    assert store.v2_pipeline is not None
    store.v2_pipeline.db.write_structured_snapshot = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    result_one = await store.consolidate(fixture["messages"], provider, "test-model")
    result_two = await store.consolidate(fixture["messages"], provider, "test-model")
    result_three = await store.consolidate(fixture["messages"], provider, "test-model")

    assert result_one is False
    assert result_two is False
    assert result_three is True
    assert store.v2_db is not None
    assert store.v2_db.list_canonical_memories() == []
    raw_events = store.v2_db.list_raw_events()
    assert len(raw_events) == 1
    assert raw_events[0]["candidate_type"] == "v2_raw_archive"
    assert "[RAW] 2 messages" in store.history_file.read_text(encoding="utf-8")
