# Phase 2 test content:
# - verifies the shadow pipeline writes structured full-snapshot sidecar artifacts into memory/shadow
# - verifies canonical memories, evidence links, rendered views, and debug payload are all emitted
# - verifies repeated replay is stable after ignoring non-deterministic event metadata
# - verifies snapshot replacement, empty-snapshot clears, and raw-archive fallback preserve correctness
# - verifies tool_choice retry and invalid snapshot failures follow the new legacy-aligned behavior
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_2/agent/test_memory_shadow_pipeline.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from nanobot.agent.memory_pipeline import ShadowMemoryPipeline
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "memory" / f"{name}.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _normalize_raw_events(events: list[dict]) -> list[dict]:
    normalized = []
    for event in events:
        normalized.append({
            "session_key": event["session_key"],
            "history_text": event["history_text"],
            "plain_text": event["plain_text"],
            "candidate_type": event["candidate_type"],
            "extracted_json": event["extracted_json"],
        })
    return normalized


def _normalize_evidence_links(links: list[dict], memories: list[dict]) -> list[tuple[str, float]]:
    memory_ids = {link["memory_id"] for link in links}
    assert memory_ids.issubset({memory["memory_id"] for memory in memories})
    return sorted((link["memory_id"], link["weight"]) for link in links)


def test_shadow_pipeline_writes_sidecar_artifacts(tmp_path: Path) -> None:
    fixture = _load_fixture("shadow_payload")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="shadow_call_1",
                    name="save_memory_shadow",
                    arguments=fixture["tool_arguments"],
                )
            ],
        )
    )

    pipeline = ShadowMemoryPipeline(tmp_path)

    result = __import__("asyncio").run(
        pipeline.ingest_chunk(fixture["messages"], provider, "test-model")
    )

    assert result is True
    assert pipeline.db.db_path.exists()
    assert pipeline.db.memory_file.read_text(encoding="utf-8") == fixture["expected_memory_markdown"]
    assert pipeline.db.history_file.read_text(encoding="utf-8") == fixture["expected_history_markdown"]

    raw_events = pipeline.db.list_raw_events()
    memories = pipeline.db.list_canonical_memories()
    evidence = pipeline.db.list_evidence_links()

    assert len(raw_events) == 1
    assert len(memories) == 2
    assert len(evidence) == 2
    assert json.loads(pipeline.debug_file.read_text(encoding="utf-8"))["history_entry"] == fixture["tool_arguments"]["history_entry"]


def test_shadow_pipeline_replay_is_stable_ignoring_event_metadata(tmp_path: Path) -> None:
    fixture = _load_fixture("shadow_payload")

    def _provider():
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="shadow_call_1",
                        name="save_memory_shadow",
                        arguments=fixture["tool_arguments"],
                    )
                ],
            )
        )
        return provider

    pipeline_one = ShadowMemoryPipeline(tmp_path / "run_one")
    pipeline_two = ShadowMemoryPipeline(tmp_path / "run_two")

    __import__("asyncio").run(pipeline_one.ingest_chunk(fixture["messages"], _provider(), "test-model"))
    __import__("asyncio").run(pipeline_two.ingest_chunk(fixture["messages"], _provider(), "test-model"))

    memories_one = pipeline_one.db.list_canonical_memories()
    memories_two = pipeline_two.db.list_canonical_memories()
    raw_one = pipeline_one.db.list_raw_events()
    raw_two = pipeline_two.db.list_raw_events()
    evidence_one = pipeline_one.db.list_evidence_links()
    evidence_two = pipeline_two.db.list_evidence_links()

    assert pipeline_one.db.memory_file.read_text(encoding="utf-8") == pipeline_two.db.memory_file.read_text(encoding="utf-8")
    assert pipeline_one.db.history_file.read_text(encoding="utf-8") == pipeline_two.db.history_file.read_text(encoding="utf-8")
    assert memories_one == memories_two
    assert _normalize_raw_events(raw_one) == _normalize_raw_events(raw_two)
    assert _normalize_evidence_links(evidence_one, memories_one) == _normalize_evidence_links(evidence_two, memories_two)


def test_shadow_pipeline_replaces_full_snapshot_when_memory_changes(tmp_path: Path) -> None:
    first_payload = {
        "history_entry": "[2026-04-01 09:00] User prefers concise answers.",
        "canonical_memories": [
            {
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "User prefers concise answers.",
                "confidence": 0.9,
            }
        ],
    }
    second_payload = {
        "history_entry": "[2026-04-01 09:05] User now prefers bullet-point concise answers.",
        "canonical_memories": [
            {
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "User prefers bullet-point concise answers.",
                "confidence": 0.93,
            }
        ],
    }

    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="shadow_call_1", name="save_memory_shadow", arguments=first_payload)]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="shadow_call_2", name="save_memory_shadow", arguments=second_payload)]),
    ])

    pipeline = ShadowMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember I prefer concise answers.", "timestamp": "2026-04-01T09:00:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:00:01"},
    ]

    __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))
    first_memories = pipeline.db.list_canonical_memories()

    __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))
    second_memories = pipeline.db.list_canonical_memories()

    assert len(first_memories) == 1
    assert len(second_memories) == 1
    assert first_memories[0]["text"] == "User prefers concise answers."
    assert second_memories[0]["text"] == "User prefers bullet-point concise answers."
    assert first_memories[0]["memory_id"] != second_memories[0]["memory_id"]


def test_shadow_pipeline_keeps_distinct_memories_with_same_subclass(tmp_path: Path) -> None:
    payload = {
        "history_entry": "[2026-04-01 09:10] User has two active projects.",
        "canonical_memories": [
            {
                "main_class": "projects",
                "sub_class": "active_project",
                "text": "The active project is nanobot.",
                "confidence": 0.91,
            },
            {
                "main_class": "projects",
                "sub_class": "active_project",
                "text": "The active project is memory-v2-eval.",
                "confidence": 0.89,
            },
        ],
    }
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="shadow_call_1", name="save_memory_shadow", arguments=payload)],
        )
    )

    pipeline = ShadowMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember the active projects are nanobot and memory-v2-eval.", "timestamp": "2026-04-01T09:10:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:10:01"},
    ]

    __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))

    memories = pipeline.db.list_canonical_memories()
    assert len(memories) == 2
    assert memories[0]["memory_id"] != memories[1]["memory_id"]
    assert {item["text"] for item in memories} == {
        "The active project is nanobot.",
        "The active project is memory-v2-eval.",
    }


def test_shadow_pipeline_empty_snapshot_clears_existing_memories(tmp_path: Path) -> None:
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="shadow_call_1",
                    name="save_memory_shadow",
                    arguments={
                        "history_entry": "[2026-04-01 09:00] User prefers concise answers.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise answers.",
                                "confidence": 0.9,
                            }
                        ],
                    },
                )
            ],
        ),
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="shadow_call_2",
                    name="save_memory_shadow",
                    arguments={
                        "history_entry": "[2026-04-01 09:05] The old preference should be removed.",
                        "canonical_memories": [],
                    },
                )
            ],
        ),
    ])

    pipeline = ShadowMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember I prefer concise answers.", "timestamp": "2026-04-01T09:00:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:00:01"},
    ]

    __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))
    __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))

    memories = pipeline.db.list_canonical_memories()
    rendered = pipeline.db.memory_file.read_text(encoding="utf-8")
    assert memories == []
    assert "(How the user prefers to communicate and collaborate)" in rendered


def test_shadow_pipeline_raw_archives_after_repeated_failures(tmp_path: Path) -> None:
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="no tool call", tool_calls=[]))

    pipeline = ShadowMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember I prefer concise answers.", "timestamp": "2026-04-01T09:00:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:00:01"},
    ]

    result_one = __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))
    result_two = __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))
    result_three = __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))

    assert result_one is False
    assert result_two is False
    assert result_three is True

    raw_events = pipeline.db.list_raw_events()
    assert len(raw_events) == 1
    assert raw_events[0]["candidate_type"] == "shadow_raw_archive"
    assert "[RAW] 2 messages" in raw_events[0]["history_text"]
    assert "Remember I prefer concise answers." in raw_events[0]["plain_text"]
    assert "[RAW] 2 messages" in pipeline.db.history_file.read_text(encoding="utf-8")


def test_shadow_pipeline_raw_archive_preserves_existing_snapshot(tmp_path: Path) -> None:
    success_provider = AsyncMock()
    success_provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="shadow_call_1",
                    name="save_memory_shadow",
                    arguments={
                        "history_entry": "[2026-04-01 09:00] User prefers concise answers.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise answers.",
                                "confidence": 0.92,
                            }
                        ],
                    },
                )
            ],
        )
    )
    failure_provider = AsyncMock()
    failure_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="no tool call", tool_calls=[]))

    pipeline = ShadowMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember I prefer concise answers.", "timestamp": "2026-04-01T09:00:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:00:01"},
    ]

    __import__("asyncio").run(pipeline.ingest_chunk(messages, success_provider, "test-model"))
    before = pipeline.db.list_canonical_memories()

    __import__("asyncio").run(pipeline.ingest_chunk(messages, failure_provider, "test-model"))
    __import__("asyncio").run(pipeline.ingest_chunk(messages, failure_provider, "test-model"))
    __import__("asyncio").run(pipeline.ingest_chunk(messages, failure_provider, "test-model"))
    after = pipeline.db.list_canonical_memories()

    assert before == after


def test_shadow_pipeline_retries_with_auto_when_forced_tool_choice_is_unsupported(tmp_path: Path) -> None:
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content='provider does not support tool_choice and should be ["none", "auto"]',
            tool_calls=[],
            finish_reason="error",
        ),
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="shadow_call_1",
                    name="save_memory_shadow",
                    arguments={
                        "history_entry": "[2026-04-01 09:20] User prefers concise answers.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise answers.",
                                "confidence": 0.94,
                            }
                        ],
                    },
                )
            ],
        ),
    ])

    pipeline = ShadowMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember I prefer concise answers.", "timestamp": "2026-04-01T09:20:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:20:01"},
    ]

    result = __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))

    assert result is True
    assert provider.chat_with_retry.await_count == 2
    first_call = provider.chat_with_retry.await_args_list[0].kwargs
    second_call = provider.chat_with_retry.await_args_list[1].kwargs
    assert isinstance(first_call["tool_choice"], dict)
    assert second_call["tool_choice"] == "auto"
    memories = pipeline.db.list_canonical_memories()
    assert len(memories) == 1
    assert memories[0]["text"] == "User prefers concise answers."


def test_shadow_pipeline_invalid_snapshot_raw_archives_after_repeated_failures(tmp_path: Path) -> None:
    success_provider = AsyncMock()
    success_provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="shadow_call_1",
                    name="save_memory_shadow",
                    arguments={
                        "history_entry": "[2026-04-01 09:00] User prefers concise answers.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise answers.",
                                "confidence": 0.92,
                            }
                        ],
                    },
                )
            ],
        )
    )
    invalid_provider = AsyncMock()
    invalid_provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="shadow_call_bad",
                    name="save_memory_shadow",
                    arguments={
                        "history_entry": "[2026-04-01 09:10] Attempted invalid snapshot update.",
                        "canonical_memories": [{"main_class": "preferences", "text": "   "}],
                    },
                )
            ],
        )
    )

    pipeline = ShadowMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember I prefer concise answers.", "timestamp": "2026-04-01T09:00:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:00:01"},
    ]

    __import__("asyncio").run(pipeline.ingest_chunk(messages, success_provider, "test-model"))
    before = pipeline.db.list_canonical_memories()

    result_one = __import__("asyncio").run(pipeline.ingest_chunk(messages, invalid_provider, "test-model"))
    result_two = __import__("asyncio").run(pipeline.ingest_chunk(messages, invalid_provider, "test-model"))
    result_three = __import__("asyncio").run(pipeline.ingest_chunk(messages, invalid_provider, "test-model"))

    after = pipeline.db.list_canonical_memories()
    raw_events = pipeline.db.list_raw_events()

    assert result_one is False
    assert result_two is False
    assert result_three is True
    assert before == after
    assert len(raw_events) == 2
    assert raw_events[-1]["candidate_type"] == "shadow_raw_archive"
    assert "[RAW] 2 messages" in raw_events[-1]["history_text"]
