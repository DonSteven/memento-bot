# Phase 2 test content:
# - verifies the structured pipeline writes root memory artifacts into memory/
# - verifies canonical memories and rendered views are emitted
# - verifies repeated replay is stable after ignoring non-deterministic event metadata
# - verifies snapshot replacement, empty-snapshot clears, and raw-archive fallback preserve correctness
# - verifies tool_choice retry, prompt seeding, and invalid snapshot failures follow the v2 behavior
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_2/agent/test_memory_shadow_pipeline.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from nanobot.agent.memory_pipeline import StructuredMemoryPipeline
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
        })
    return normalized


def test_structured_pipeline_writes_root_artifacts(tmp_path: Path) -> None:
    fixture = _load_fixture("shadow_payload")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="v2_call_1",
                    name="save_memory_structured",
                    arguments=fixture["tool_arguments"],
                )
            ],
        )
    )

    pipeline = StructuredMemoryPipeline(tmp_path)

    result = __import__("asyncio").run(
        pipeline.ingest_chunk(fixture["messages"], provider, "test-model")
    )

    assert result is True
    assert pipeline.db.db_path.exists()
    assert pipeline.db.memory_file.read_text(encoding="utf-8") == fixture["expected_memory_markdown"]
    assert pipeline.db.history_file.read_text(encoding="utf-8") == fixture["expected_history_markdown"]

    raw_events = pipeline.db.list_raw_events()
    memories = pipeline.db.list_canonical_memories()

    assert len(raw_events) == 1
    assert raw_events[0]["candidate_type"] == "v2_snapshot"
    assert len(memories) == 2
    assert not (tmp_path / "memory" / "last_payload.json").exists()
    assert not (tmp_path / "memory" / "shadow").exists()


def test_structured_pipeline_replay_is_stable_ignoring_event_metadata(tmp_path: Path) -> None:
    fixture = _load_fixture("shadow_payload")

    def _provider():
        provider = AsyncMock()
        provider.chat_with_retry = AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="v2_call_1",
                        name="save_memory_structured",
                        arguments=fixture["tool_arguments"],
                    )
                ],
            )
        )
        return provider

    pipeline_one = StructuredMemoryPipeline(tmp_path / "run_one")
    pipeline_two = StructuredMemoryPipeline(tmp_path / "run_two")

    __import__("asyncio").run(pipeline_one.ingest_chunk(fixture["messages"], _provider(), "test-model"))
    __import__("asyncio").run(pipeline_two.ingest_chunk(fixture["messages"], _provider(), "test-model"))

    memories_one = pipeline_one.db.list_canonical_memories()
    memories_two = pipeline_two.db.list_canonical_memories()
    raw_one = pipeline_one.db.list_raw_events()
    raw_two = pipeline_two.db.list_raw_events()

    assert pipeline_one.db.memory_file.read_text(encoding="utf-8") == pipeline_two.db.memory_file.read_text(encoding="utf-8")
    assert pipeline_one.db.history_file.read_text(encoding="utf-8") == pipeline_two.db.history_file.read_text(encoding="utf-8")
    assert memories_one == memories_two
    assert _normalize_raw_events(raw_one) == _normalize_raw_events(raw_two)


def test_structured_pipeline_builds_consolidation_prompt_from_db_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "MEMORY.md").write_text(
        "# Long-term Memory\n\n## Preferences\n\n- reply_style: stale file content that should not be reused.",
        encoding="utf-8",
    )

    pipeline = StructuredMemoryPipeline(workspace)
    pipeline.db.replace_canonical_snapshot([
        {
            "memory_id": "mem_active",
            "main_class": "preferences",
            "sub_class": "reply_style",
            "text": "User prefers concise answers.",
        },
    ])

    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="v2_call_1",
                    name="save_memory_structured",
                    arguments={
                        "history_entry": "[2026-04-01 09:00] User prefers concise answers.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise answers.",
                            }
                        ],
                    },
                )
            ],
        )
    )

    __import__("asyncio").run(
        pipeline.ingest_chunk(
            [
                {"role": "user", "content": "Remember I prefer concise answers.", "timestamp": "2026-04-01T09:00:00"},
                {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:00:01"},
            ],
            provider,
            "test-model",
        )
    )

    prompt = provider.chat_with_retry.await_args_list[0].kwargs["messages"][1]["content"]
    assert "## Preferences" in prompt
    assert "reply_style: User prefers concise answers." in prompt
    assert "## Inactive Memory Ledger" not in prompt
    assert "stale file content that should not be reused" not in prompt


def test_structured_pipeline_build_retrieval_context_returns_empty_without_db(tmp_path: Path) -> None:
    pipeline = StructuredMemoryPipeline(tmp_path)

    assert pipeline.build_retrieval_context("What should I know?") == ""


def test_structured_pipeline_build_retrieval_context_returns_core_memory_for_empty_query(tmp_path: Path) -> None:
    pipeline = StructuredMemoryPipeline(tmp_path)
    pipeline.db.upsert_canonical_memory(
        memory_id="mem_profile",
        main_class="personal_profile",
        sub_class="environment",
        text="User works on Linux.",
    )
    pipeline.db.upsert_canonical_memory(
        memory_id="mem_pref",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise answers.",
    )
    pipeline.db.upsert_canonical_memory(
        memory_id="mem_project",
        main_class="projects",
        sub_class="active_project",
        text="The active project is nanobot.",
    )

    context = pipeline.build_retrieval_context("   ")

    assert "## Core Memory" in context
    assert "[personal_profile/environment] User works on Linux." in context
    assert "[preferences/reply_style] User prefers concise answers." in context
    assert "## Retrieved Memory" not in context
    assert "The active project is nanobot." not in context


def test_structured_pipeline_build_retrieval_context_appends_non_core_hits(tmp_path: Path) -> None:
    pipeline = StructuredMemoryPipeline(tmp_path)
    pipeline.db.upsert_canonical_memory(
        memory_id="mem_profile",
        main_class="personal_profile",
        sub_class="environment",
        text="User works on Linux.",
    )
    pipeline.db.upsert_canonical_memory(
        memory_id="mem_project",
        main_class="projects",
        sub_class="active_project",
        text="The active project is nanobot.",
    )

    context = pipeline.build_retrieval_context("What's the active project?")

    assert "## Core Memory" in context
    assert "## Retrieved Memory" in context
    assert "[projects/active_project] The active project is nanobot." in context


def test_structured_pipeline_build_retrieval_context_dedupes_matches_without_memory_id(tmp_path: Path) -> None:
    pipeline = StructuredMemoryPipeline(tmp_path)
    pipeline.db.initialize()
    with pipeline.db.connect() as conn:
        conn.execute(
            """
            insert into canonical_memories(memory_id, main_class, sub_class, text)
            values (?, ?, ?, ?)
            """,
            ("mem_pref", "preferences", "reply_style", "User prefers concise answers."),
        )
        conn.execute(
            """
            insert into canonical_fts(memory_id, main_class, sub_class, text)
            values (?, ?, ?, ?)
            """,
            ("mem_pref", "preferences", "reply_style", "User prefers concise answers."),
        )
        conn.execute(
            """
            insert into canonical_fts(memory_id, main_class, sub_class, text)
            values (?, ?, ?, ?)
            """,
            ("", "preferences", "reply_style", "User prefers concise answers."),
        )

    context = pipeline.build_retrieval_context("concise answers")

    assert context.count("[preferences/reply_style] User prefers concise answers.") == 1


def test_structured_pipeline_replaces_full_snapshot_when_memory_changes(tmp_path: Path) -> None:
    first_payload = {
        "history_entry": "[2026-04-01 09:00] User prefers concise answers.",
        "canonical_memories": [
            {
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "User prefers concise answers.",
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
            }
        ],
    }

    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="v2_call_1", name="save_memory_structured", arguments=first_payload)]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="v2_call_2", name="save_memory_structured", arguments=second_payload)]),
    ])

    pipeline = StructuredMemoryPipeline(tmp_path)
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


def test_structured_pipeline_keeps_distinct_memories_with_same_subclass(tmp_path: Path) -> None:
    payload = {
        "history_entry": "[2026-04-01 09:10] User has two active projects.",
        "canonical_memories": [
            {
                "main_class": "projects",
                "sub_class": "active_project",
                "text": "The active project is nanobot.",
            },
            {
                "main_class": "projects",
                "sub_class": "active_project",
                "text": "The active project is memory-v2-eval.",
            },
        ],
    }
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="v2_call_1", name="save_memory_structured", arguments=payload)],
        )
    )

    pipeline = StructuredMemoryPipeline(tmp_path)
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


def test_structured_pipeline_accepts_new_subclass_names(tmp_path: Path) -> None:
    payload = {
        "history_entry": "[2026-04-01 09:11] User prefers high-signal implementation summaries.",
        "canonical_memories": [
            {
                "main_class": "preferences",
                "sub_class": "implementation_summary_style",
                "text": "User prefers high-signal implementation summaries.",
            }
        ],
    }
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="v2_call_1", name="save_memory_structured", arguments=payload)],
        )
    )

    pipeline = StructuredMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember I prefer high-signal implementation summaries.", "timestamp": "2026-04-01T09:11:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:11:01"},
    ]

    __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))

    memories = pipeline.db.list_canonical_memories()
    assert memories == [
        {
            "memory_id": memories[0]["memory_id"],
            "main_class": "preferences",
            "sub_class": "implementation_summary_style",
            "text": "User prefers high-signal implementation summaries.",
        }
    ]
    assert "implementation_summary_style: User prefers high-signal implementation summaries." in (
        pipeline.db.memory_file.read_text(encoding="utf-8")
    )


def test_structured_pipeline_persists_all_memories_in_memory_view(tmp_path: Path) -> None:
    payload = {
        "history_entry": "[2026-04-01 09:12] User prefers both concise and verbose answers in different contexts.",
        "canonical_memories": [
            {
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "User prefers concise answers.",
            },
            {
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "User once preferred verbose answers.",
            },
        ],
    }
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id="v2_call_1", name="save_memory_structured", arguments=payload)],
        )
    )

    pipeline = StructuredMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember I now prefer concise answers.", "timestamp": "2026-04-01T09:12:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:12:01"},
    ]

    __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))

    memories = pipeline.db.list_canonical_memories()
    rendered = pipeline.db.memory_file.read_text(encoding="utf-8")
    queried = pipeline.db.query_canonical_memories("verbose", limit=5)

    assert len(memories) == 2
    assert "User prefers concise answers." in rendered
    assert "User once preferred verbose answers." in rendered
    assert len(queried) == 1
    assert queried[0]["text"] == "User once preferred verbose answers."


def test_structured_pipeline_consolidation_view_matches_rendered_memory_view(tmp_path: Path) -> None:
    pipeline = StructuredMemoryPipeline(tmp_path)
    pipeline.db.replace_canonical_snapshot([
        {
            "memory_id": "mem_active",
            "main_class": "preferences",
            "sub_class": "reply_style",
            "text": "User prefers concise answers.",
        }
    ])

    view = pipeline._build_consolidation_memory_view()

    assert view == pipeline.db.render_memory_view()
    assert "## Preferences" in view
    assert "reply_style: User prefers concise answers." in view


def test_structured_pipeline_consolidation_view_includes_all_snapshot_memories(tmp_path: Path) -> None:
    pipeline = StructuredMemoryPipeline(tmp_path)
    pipeline.db.replace_canonical_snapshot([
        {
            "memory_id": "mem_active",
            "main_class": "preferences",
            "sub_class": "reply_style",
            "text": "User prefers concise answers.",
        },
        {
            "memory_id": "mem_verbose",
            "main_class": "preferences",
            "sub_class": "reply_style",
            "text": "User once preferred verbose answers.",
        },
        {
            "memory_id": "mem_project",
            "main_class": "projects",
            "sub_class": "active_project",
            "text": "The active project is nanobot.",
        },
    ])

    view = pipeline._build_consolidation_memory_view()

    assert "## Preferences" in view
    assert "reply_style: User prefers concise answers." in view
    assert "reply_style: User once preferred verbose answers." in view
    assert "## Projects" in view
    assert "active_project: The active project is nanobot." in view


def test_structured_pipeline_empty_snapshot_clears_existing_memories(tmp_path: Path) -> None:
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="v2_call_1",
                    name="save_memory_structured",
                    arguments={
                        "history_entry": "[2026-04-01 09:00] User prefers concise answers.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise answers.",
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
                    id="v2_call_2",
                    name="save_memory_structured",
                    arguments={
                        "history_entry": "[2026-04-01 09:05] The old preference should be removed.",
                        "canonical_memories": [],
                    },
                )
            ],
        ),
    ])

    pipeline = StructuredMemoryPipeline(tmp_path)
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


def test_structured_pipeline_second_consolidation_prompt_preserves_existing_snapshot_context(tmp_path: Path) -> None:
    first_payload = {
        "history_entry": "[2026-04-01 09:00] User prefers concise and verbose answers in different contexts.",
        "canonical_memories": [
            {
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "User prefers concise answers.",
            },
            {
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "User once preferred verbose answers.",
            },
        ],
    }
    second_payload = {
        "history_entry": "[2026-04-01 09:05] Existing preference context remains available.",
        "canonical_memories": [
            {
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "User prefers concise answers.",
            },
            {
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "User once preferred verbose answers.",
            },
        ],
    }

    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="v2_call_1", name="save_memory_structured", arguments=first_payload)]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id="v2_call_2", name="save_memory_structured", arguments=second_payload)]),
    ])

    pipeline = StructuredMemoryPipeline(tmp_path)
    messages = [
        {"role": "user", "content": "Remember I prefer concise answers now.", "timestamp": "2026-04-01T09:00:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T09:00:01"},
    ]

    __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))
    __import__("asyncio").run(pipeline.ingest_chunk(messages, provider, "test-model"))

    second_prompt = provider.chat_with_retry.await_args_list[1].kwargs["messages"][1]["content"]
    memories = pipeline.db.list_canonical_memories()

    assert "## Preferences" in second_prompt
    assert "reply_style: User once preferred verbose answers." in second_prompt
    assert len(memories) == 2


def test_structured_pipeline_raw_archives_after_repeated_failures(tmp_path: Path) -> None:
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="no tool call", tool_calls=[]))

    pipeline = StructuredMemoryPipeline(tmp_path)
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
    assert raw_events[0]["candidate_type"] == "v2_raw_archive"
    assert "[RAW] 2 messages" in raw_events[0]["history_text"]
    assert "Remember I prefer concise answers." in raw_events[0]["plain_text"]
    assert "[RAW] 2 messages" in pipeline.db.history_file.read_text(encoding="utf-8")


def test_structured_pipeline_raw_archive_preserves_existing_snapshot(tmp_path: Path) -> None:
    success_provider = AsyncMock()
    success_provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="v2_call_1",
                    name="save_memory_structured",
                    arguments={
                        "history_entry": "[2026-04-01 09:00] User prefers concise answers.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise answers.",
                            }
                        ],
                    },
                )
            ],
        )
    )
    failure_provider = AsyncMock()
    failure_provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="no tool call", tool_calls=[]))

    pipeline = StructuredMemoryPipeline(tmp_path)
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


def test_structured_pipeline_retries_with_auto_when_forced_tool_choice_is_unsupported(tmp_path: Path) -> None:
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
                    id="v2_call_1",
                    name="save_memory_structured",
                    arguments={
                        "history_entry": "[2026-04-01 09:20] User prefers concise answers.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise answers.",
                            }
                        ],
                    },
                )
            ],
        ),
    ])

    pipeline = StructuredMemoryPipeline(tmp_path)
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


def test_structured_pipeline_invalid_snapshot_raw_archives_after_repeated_failures(tmp_path: Path) -> None:
    success_provider = AsyncMock()
    success_provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="v2_call_1",
                    name="save_memory_structured",
                    arguments={
                        "history_entry": "[2026-04-01 09:00] User prefers concise answers.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise answers.",
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
                    id="v2_call_bad",
                    name="save_memory_structured",
                    arguments={
                        "history_entry": "[2026-04-01 09:10] Attempted invalid snapshot update.",
                        "canonical_memories": [{"main_class": "preferences", "text": "User prefers concise answers."}],
                    },
                )
            ],
        )
    )

    pipeline = StructuredMemoryPipeline(tmp_path)
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
    assert raw_events[-1]["candidate_type"] == "v2_raw_archive"
    assert "[RAW] 2 messages" in raw_events[-1]["history_text"]
