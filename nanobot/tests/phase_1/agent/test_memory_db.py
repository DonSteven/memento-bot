# Phase 1 test content:
# - verifies the SQLite storage base initializes idempotently
# - verifies raw events, canonical memories, and evidence links can be stored and read back
# - verifies canonical memory upsert updates an existing record instead of duplicating it
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
        priority=0.8,
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
        main_class="project_context",
        sub_class="active_project",
        text="Active project is alpha.",
        version=1,
    )
    db.upsert_canonical_memory(
        memory_id="mem_1",
        main_class="project_context",
        sub_class="active_project",
        text="Active project is beta.",
        version=2,
    )

    memories = db.list_canonical_memories()
    assert len(memories) == 1
    assert memories[0]["text"] == "Active project is beta."
    assert memories[0]["version"] == 2
