# Phase 1 test content:
# - verifies the SQLite storage base initializes idempotently
# - verifies raw events, canonical memories, and evidence links can be stored and read back
# - verifies canonical memory upsert updates an existing record instead of duplicating it
# - verifies full snapshot replacement can regenerate distinct memory_id values from content
# - verifies canonical memory reads expose only the active snapshot fields
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_1/agent/test_memory_db.py

from __future__ import annotations

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
        confidence=0.9,
    )
    db.upsert_canonical_memory(
        memory_id="mem_1",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise replies.",
        confidence=0.95,
    )
    db.add_evidence_link(memory_id="mem_1", event_id="evt_1", weight=1.0)

    raw_events = db.list_raw_events()
    memories = db.list_canonical_memories()
    evidence = db.list_evidence_links()

    assert len(raw_events) == 1
    assert raw_events[0]["event_id"] == "evt_1"
    assert len(memories) == 1
    assert memories[0]["memory_id"] == "mem_1"
    assert memories[0]["text"] == "User prefers concise replies."
    assert len(evidence) == 1
    assert evidence[0] == {"memory_id": "mem_1", "event_id": "evt_1", "weight": 1.0}


def test_upsert_canonical_memory_replaces_existing_row(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    db.upsert_canonical_memory(
        memory_id="mem_1",
        main_class="projects",
        sub_class="active_project",
        text="Active project is alpha.",
        confidence=0.1,
    )
    db.upsert_canonical_memory(
        memory_id="mem_1",
        main_class="projects",
        sub_class="active_project",
        text="Active project is beta.",
        confidence=0.8,
    )

    memories = db.list_canonical_memories()
    assert len(memories) == 1
    assert memories[0]["text"] == "Active project is beta."
    assert memories[0]["confidence"] == 0.8


def test_replace_canonical_snapshot_generates_distinct_ids_per_text(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    db.replace_canonical_snapshot([
        {
            "main_class": "projects",
            "sub_class": "active_project",
            "text": "The active project is nanobot.",
            "confidence": 0.9,
        },
        {
            "main_class": "projects",
            "sub_class": "active_project",
            "text": "The active project is memory-v2-eval.",
            "confidence": 0.88,
        },
    ])

    memories = db.list_canonical_memories()

    assert len(memories) == 2
    assert memories[0]["memory_id"] != memories[1]["memory_id"]
    assert {item["text"] for item in memories} == {
        "The active project is nanobot.",
        "The active project is memory-v2-eval.",
    }


def test_canonical_memory_reads_return_active_snapshot_fields_only(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)

    db.upsert_canonical_memory(
        memory_id="mem_pref",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise replies.",
        confidence=0.91,
    )

    listed = db.list_canonical_memories()
    queried = db.query_canonical_memories("concise replies", limit=5)

    expected_keys = {"memory_id", "main_class", "sub_class", "text", "confidence"}
    assert listed == [
        {
            "memory_id": "mem_pref",
            "main_class": "preferences",
            "sub_class": "reply_style",
            "text": "User prefers concise replies.",
            "confidence": 0.91,
        }
    ]
    assert set(listed[0]) == expected_keys
    assert len(queried) == 1
    assert set(queried[0]) == expected_keys
