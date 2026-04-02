# Phase 3 test content:
# - verifies v2 mode persists memory through memory.db and re-renders the legacy MEMORY.md/HISTORY.md views
# - verifies parsed canonical memories and evidence links are written into the database
# - verifies v2 falls back to legacy file writes when database-backed persistence fails
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
                name="save_memory",
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
    evidence = store.v2_db.list_evidence_links()

    assert [item["main_class"] for item in canonical] == fixture["expected"]["main_classes"]
    assert [item["sub_class"] for item in canonical] == fixture["expected"]["sub_classes"]
    assert [item["text"] for item in canonical] == fixture["expected"]["texts"]
    assert len(raw_events) == 1
    assert raw_events[0]["history_text"] == fixture["tool_arguments"]["history_entry"]
    assert len(evidence) == len(canonical)


@pytest.mark.asyncio
async def test_v2_mode_falls_back_to_legacy_files_on_db_failure(tmp_path: Path) -> None:
    fixture = _load_fixture("v2_payload")
    store = MemoryStore(tmp_path, mode="v2")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=_provider_response(fixture["tool_arguments"]))
    assert store.v2_db is not None
    store.v2_db.replace_canonical_snapshot = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    result = await store.consolidate(fixture["messages"], provider, "test-model")

    assert result is True
    assert store.memory_file.read_text(encoding="utf-8") == fixture["tool_arguments"]["memory_update"]
    assert fixture["tool_arguments"]["history_entry"] in store.history_file.read_text(encoding="utf-8")
