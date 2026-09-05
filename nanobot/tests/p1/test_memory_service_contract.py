from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.memory_db import MemoryContext, MemoryRecord, MemorySnapshot
from nanobot.agent.memory_pipeline import StructuredMemoryPipeline
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionManager
from nanobot.tests.memory_test_utils import TestMemoryService as MemoryService


def _provider(arguments: dict | None = None):
    provider = AsyncMock()
    if arguments is not None:
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-1", name="save_memory_structured", arguments=arguments)
                ],
            )
        )
    return provider


def _seed(service: MemoryService) -> None:
    classes = (
        "personal_profile",
        "preferences",
        "constraints",
        "projects",
        "daily_life",
        "plans_commitments",
    )
    records = tuple(
        MemoryRecord(
            f"{main_class}-{index}",
            main_class,
            f"slot-{index}",
            f"{main_class} nanobot fact {index}",
        )
        for main_class in classes
        for index in range(2)
    )
    db = service.database
    db.commit_snapshot(
        MemorySnapshot(0, records),
        expected_revision=0,
        event_id="seed",
        ts="2026-09-05T00:00:00",
        session_key="test",
        history_text="seed",
        dynamic_embeddings={
            item.memory_id: [1.0, 0.0, 0.0]
            for item in records
            if item.main_class in {"projects", "daily_life", "plans_commitments"}
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "completely unrelated query"])
async def test_core_memory_is_complete_once_for_empty_and_unrelated_queries(
    tmp_path, query
) -> None:
    service = MemoryService(tmp_path, _provider(), "test-model")
    _seed(service)

    context = await service.prepare_context(query, retrieval_budget=2)

    assert len(context.core_items) == 6
    assert len({item.memory_id for item in context.core_items}) == 6
    assert context.retrieved_items == ()
    assert context.render().count("## Core Memory") == 1


@pytest.mark.asyncio
async def test_dynamic_top_k_does_not_include_or_duplicate_core_memory(tmp_path) -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "retrieval_contract.json").read_text(encoding="utf-8")
    )
    service = MemoryService(tmp_path, _provider(), "test-model")
    _seed(service)

    context = await service.prepare_context(
        fixture["query"],
        retrieval_budget=fixture["retrieval_budget"],
    )

    assert [item.memory_id for item in context.core_items] == fixture["expected"]["core_ids"]
    assert [item.memory_id for item in context.retrieved_items] == fixture["expected"][
        "retrieved_ids"
    ]
    assert all(
        item.main_class in set(fixture["dynamic_classes"]) for item in context.retrieved_items
    )
    assert not (
        {item.memory_id for item in context.core_items}
        & {item.memory_id for item in context.retrieved_items}
    )


@pytest.mark.asyncio
async def test_pipeline_returns_complete_snapshot_without_storage_side_effects(tmp_path) -> None:
    arguments = {
        "history_entry": "[2026-09-05 10:00] captured two project facts",
        "canonical_memories": [
            {"main_class": "projects", "sub_class": "active", "text": "Project alpha"},
            {"main_class": "projects", "sub_class": "active", "text": "Project beta"},
        ],
    }
    provider = _provider(arguments)
    pipeline = StructuredMemoryPipeline(provider, "test-model")

    snapshot, history = await pipeline.extract_snapshot(
        [{"role": "user", "content": "Remember alpha and beta"}],
        MemorySnapshot(7),
    )

    assert snapshot.revision == 7
    assert {item.text for item in snapshot.memories} == {"Project alpha", "Project beta"}
    assert history.startswith("[2026-09-05")
    assert not (tmp_path / "memory").exists()


@pytest.mark.asyncio
async def test_committed_snapshot_reports_view_export_failure_without_losing_data(
    tmp_path, monkeypatch
) -> None:
    provider = _provider(
        {
            "history_entry": "[2026-09-05 10:00] saved",
            "canonical_memories": [
                {"main_class": "preferences", "sub_class": "style", "text": "Be concise"},
            ],
        }
    )
    service = MemoryService(tmp_path, provider, "test-model")
    await service.sync_markdown()
    monkeypatch.setattr(
        service.synchronizer,
        "publish_pending",
        lambda: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = await service.consolidate(
        [{"role": "user", "content": "Be concise"}],
        session_key="cli:test",
    )

    assert result.database_committed is True
    assert result.view_exported is False
    assert result.revision == 1
    assert service.database.read_snapshot().memories[0].text == "Be concise"
    assert len(service.database.list_raw_events()) == 1


def test_context_builder_only_uses_prepared_memory_context(tmp_path) -> None:
    builder = ContextBuilder(tmp_path)
    prepared = MemoryContext(
        core_items=(MemoryRecord("id", "preferences", "style", "Be concise"),),
    )
    messages = builder.build_messages([], "hello", memory_context=prepared)
    assert "[preferences/style] Be concise" in messages[0]["content"]


@pytest.mark.asyncio
async def test_pipeline_retries_with_auto_when_forced_tool_choice_is_unsupported(tmp_path) -> None:
    valid = {
        "history_entry": "[2026-09-05 10:00] saved",
        "canonical_memories": [],
    }
    provider = _provider()
    provider.chat_with_retry = AsyncMock(
        side_effect=[
            LLMResponse(content="tool_choice does not support forced calls", finish_reason="error"),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(id="call-2", name="save_memory_structured", arguments=valid)
                ],
            ),
        ]
    )
    pipeline = StructuredMemoryPipeline(provider, "test-model")

    snapshot, _ = await pipeline.extract_snapshot(
        [{"role": "user", "content": "nothing to remember"}],
        MemorySnapshot(2),
    )

    assert snapshot == MemorySnapshot(2)
    assert provider.chat_with_retry.await_count == 2
    assert provider.chat_with_retry.await_args_list[1].kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_repeated_extraction_failure_raw_archives_without_changing_snapshot(tmp_path) -> None:
    provider = _provider()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="no tool call"))
    service = MemoryService(tmp_path, provider, "test-model")
    _seed(service)
    before_ids = {item.memory_id for item in service.database.read_snapshot().memories}
    consolidator = MemoryConsolidator(
        service,
        provider,
        "test-model",
        SessionManager(tmp_path),
        4096,
        MagicMock(return_value=[]),
        MagicMock(return_value=[]),
    )

    result = await consolidator.archive_messages(
        [{"role": "user", "content": "archive me"}],
        session_key="cli:test",
    )

    assert result.database_committed and result.view_exported
    assert provider.chat_with_retry.await_count == 3
    assert service.database.list_raw_events()[-1]["candidate_type"] == "raw_archive"
    assert {item.memory_id for item in service.database.read_snapshot().memories} == before_ids


@pytest.mark.asyncio
async def test_committed_view_failure_is_not_re_extracted(tmp_path, monkeypatch) -> None:
    provider = _provider({"history_entry": "[2026-09-05 10:00] saved", "canonical_memories": []})
    service = MemoryService(tmp_path, provider, "test-model")
    await service.sync_markdown()
    monkeypatch.setattr(
        service.synchronizer,
        "publish_pending",
        lambda: (_ for _ in ()).throw(OSError("disk full")),
    )
    consolidator = MemoryConsolidator(
        service,
        provider,
        "test-model",
        SessionManager(tmp_path),
        4096,
        MagicMock(return_value=[]),
        MagicMock(return_value=[]),
    )

    result = await consolidator.archive_messages(
        [{"role": "user", "content": "archive me"}],
        session_key="cli:test",
    )

    assert result.database_committed is True and result.view_exported is False
    assert provider.chat_with_retry.await_count == 1
    assert len(service.database.list_raw_events()) == 1
