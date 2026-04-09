# Phase 1 test content:
# - verifies the SQLite storage base initializes idempotently
# - verifies raw events and canonical memories can be stored and read back
# - verifies canonical memory upsert updates an existing record instead of duplicating it
# - verifies full snapshot replacement can regenerate distinct memory_id values from content
# - verifies canonical memory reads expose the simplified snapshot fields
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_1/agent/test_memory_db.py

from __future__ import annotations

import sqlite3

import pytest

from nanobot.agent.memory_db import MemoryDatabase, SCHEMA_VERSION


def test_initialize_is_idempotent(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    db.initialize()
    db.initialize()

    assert db.db_path.exists()
    assert db.get_schema_version() == SCHEMA_VERSION


def test_store_and_list_records(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    db.insert_raw_event(
        event_id="evt_1",
        ts="2026-04-01T10:00:00",
        session_key="cli:test",
        history_text="[2026-04-01 10:00] User prefers concise replies.",
        plain_text="User said they prefer concise replies.",
        main_class="preferences",
        sub_class="reply_style",
        candidate_type="fact",
    )
    db.upsert_canonical_memory(
        memory_id="mem_1",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise replies.",
    )
    raw_events = db.list_raw_events()
    memories = db.list_canonical_memories()

    assert len(raw_events) == 1
    assert raw_events[0]["event_id"] == "evt_1"
    assert "extracted_json" not in raw_events[0]
    assert len(memories) == 1
    assert memories[0]["memory_id"] == "mem_1"
    assert memories[0]["text"] == "User prefers concise replies."


def test_initialize_rejects_legacy_raw_events_schema_with_extracted_json(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    with sqlite3.connect(db.db_path) as conn:
        conn.executescript(
            """
            create table raw_events (
                event_id text primary key,
                ts text not null,
                session_key text not null,
                history_text text not null,
                plain_text text not null default '',
                extracted_json text,
                main_class text,
                sub_class text,
                candidate_type text,
                    source_start_idx integer,
                    source_end_idx integer
                );
            """
        )

    with pytest.raises(RuntimeError, match="Delete memory/memory.db manually"):
        db.initialize()


def test_upsert_canonical_memory_replaces_existing_row(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    db.upsert_canonical_memory(
        memory_id="mem_1",
        main_class="projects",
        sub_class="active_project",
        text="Active project is alpha.",
    )
    db.upsert_canonical_memory(
        memory_id="mem_1",
        main_class="projects",
        sub_class="active_project",
        text="Active project is beta.",
    )

    memories = db.list_canonical_memories()
    assert len(memories) == 1
    assert memories[0]["text"] == "Active project is beta."


def test_upsert_canonical_memory_rejects_blank_subclass(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    with pytest.raises(ValueError, match="sub_class must be a non-empty string"):
        db.upsert_canonical_memory(
            memory_id="mem_1",
            main_class="projects",
            sub_class="   ",
            text="Active project is alpha.",
        )


def test_replace_canonical_snapshot_generates_distinct_ids_per_text(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    db.replace_canonical_snapshot([
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
    ])

    memories = db.list_canonical_memories()

    assert len(memories) == 2
    assert memories[0]["memory_id"] != memories[1]["memory_id"]
    assert {item["text"] for item in memories} == {
        "The active project is nanobot.",
        "The active project is memory-v2-eval.",
    }


def test_replace_canonical_snapshot_rejects_blank_subclass(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    with pytest.raises(ValueError, match="sub_class must be a non-empty string"):
        db.replace_canonical_snapshot([
            {
                "main_class": "projects",
                "sub_class": "",
                "text": "The active project is nanobot.",
            }
        ])


def test_canonical_memory_reads_return_snapshot_fields_only(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    db.upsert_canonical_memory(
        memory_id="mem_pref",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise replies.",
    )

    listed = db.list_canonical_memories()
    queried = db.query_canonical_memories("concise replies", limit=5)

    expected_keys = {"memory_id", "main_class", "sub_class", "text"}
    assert listed == [
        {
            "memory_id": "mem_pref",
            "main_class": "preferences",
            "sub_class": "reply_style",
            "text": "User prefers concise replies.",
        }
    ]
    assert set(listed[0]) == expected_keys
    assert len(queried) == 1
    assert set(queried[0]) == expected_keys


def test_render_and_query_expose_all_memories(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    db.replace_canonical_snapshot([
        {
            "memory_id": "mem_active",
            "main_class": "preferences",
            "sub_class": "reply_style",
            "text": "User prefers concise replies.",
        },
        {
            "memory_id": "mem_verbose",
            "main_class": "preferences",
            "sub_class": "reply_style",
            "text": "User once preferred verbose replies.",
        },
    ])

    rendered = db.render_memory_view()
    queried = db.query_canonical_memories("verbose", limit=5)

    assert "User prefers concise replies." in rendered
    assert "User once preferred verbose replies." in rendered
    assert len(queried) == 1
    assert queried[0]["text"] == "User once preferred verbose replies."
