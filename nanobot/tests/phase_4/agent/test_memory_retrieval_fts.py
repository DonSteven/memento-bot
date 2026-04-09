# Phase 4 test content:
# - verifies v2 canonical memories are queryable through SQLite FTS5
# - verifies retrieval returns relevant canonical memories and respects empty-query behavior
# - verifies retrieval results expose the simplified canonical snapshot fields
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_4/agent/test_memory_retrieval_fts.py

from __future__ import annotations

from nanobot.agent.memory_db import MemoryDatabase


def test_query_canonical_memories_returns_relevant_matches(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.upsert_canonical_memory(
        memory_id="mem_pref",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise answers.",
    )
    db.upsert_canonical_memory(
        memory_id="mem_project",
        main_class="projects",
        sub_class="active_project",
        text="The active project is nanobot.",
    )

    results = db.query_canonical_memories("concise answers", limit=5)

    assert len(results) == 1
    assert results[0]["memory_id"] == "mem_pref"
    assert results[0]["text"] == "User prefers concise answers."


def test_query_canonical_memories_empty_query_returns_no_matches(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.upsert_canonical_memory(
        memory_id="mem_pref",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise answers.",
    )

    assert db.query_canonical_memories("   ", limit=5) == []


def test_query_canonical_memories_handles_simple_word_variants(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.upsert_canonical_memory(
        memory_id="mem_pref",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise answers.",
    )

    results = db.query_canonical_memories("Please answer concisely.", limit=5)

    assert len(results) == 1
    assert results[0]["memory_id"] == "mem_pref"


def test_query_canonical_memories_returns_snapshot_fields_only(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.upsert_canonical_memory(
        memory_id="mem_pref",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise answers.",
    )

    results = db.query_canonical_memories("concise answers", limit=5)

    assert len(results) == 1
    assert results[0] == {
        "memory_id": "mem_pref",
        "main_class": "preferences",
        "sub_class": "reply_style",
        "text": "User prefers concise answers.",
    }


def test_query_canonical_memories_returns_all_matching_memories(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.upsert_canonical_memory(
        memory_id="mem_active",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise answers.",
    )
    db.upsert_canonical_memory(
        memory_id="mem_verbose",
        main_class="preferences",
        sub_class="reply_style",
        text="User once preferred verbose answers.",
    )

    results = db.query_canonical_memories("verbose", limit=5)

    assert len(results) == 1
    assert results[0]["memory_id"] == "mem_verbose"
