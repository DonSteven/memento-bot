from __future__ import annotations

import sqlite3

import pytest

from nanobot.agent.memory_db import (
    MemoryDatabase,
    MemoryRecord,
    MemoryRevisionConflictError,
    MemorySnapshot,
)


def _record(memory_id: str, main_class: str, text: str, sub_class: str = "fact") -> MemoryRecord:
    return MemoryRecord(memory_id, main_class, sub_class, text)


def _commit(db: MemoryDatabase, records: tuple[MemoryRecord, ...], revision: int = 0) -> int:
    db.initialize(3, embedding_provider="dashscope", embedding_model="fixed-test")
    return db.commit_snapshot(
        MemorySnapshot(revision, records),
        expected_revision=revision,
        event_id=f"event-{revision}",
        ts=f"2026-09-05T00:00:0{revision}",
        session_key="cli:test",
        history_text=f"history {revision}",
        dynamic_embeddings={
            item.memory_id: [1.0, 0.0, 0.0]
            for item in records
            if item.main_class in {"projects", "daily_life", "plans_commitments"}
        },
    )


def test_snapshot_commit_writes_event_memory_fts_and_revision_atomically(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    records = (
        _record("project-1", "projects", "Nanobot memory implementation"),
        _record("preference-1", "preferences", "Use concise answers"),
    )

    assert _commit(db, records) == 1
    snapshot = db.read_snapshot()

    assert snapshot.revision == 1
    assert set(snapshot.memories) == set(records)
    assert [item.memory_id for item in db.query_dynamic_memories("nanobot", 5)] == ["project-1"]
    assert [event["event_id"] for event in db.list_raw_events()] == ["event-0"]


def test_failed_snapshot_commit_leaves_no_partial_event_or_revision(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    duplicate_id = (
        _record("same", "projects", "first"),
        _record("same", "daily_life", "second"),
    )

    with pytest.raises(sqlite3.IntegrityError):
        _commit(db, duplicate_id)

    assert db.read_snapshot() == MemorySnapshot(0)
    assert db.list_raw_events() == []


def test_expected_revision_prevents_stale_snapshot_overwrite(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    _commit(db, (_record("first", "projects", "first project"),))

    with pytest.raises(MemoryRevisionConflictError, match="expected 0, found 1"):
        _commit(db, (_record("stale", "projects", "stale project"),), revision=0)

    assert [item.memory_id for item in db.read_snapshot().memories] == ["first"]
    assert len(db.list_raw_events()) == 1


def test_repeated_queries_do_not_change_data_revision_or_fts(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    _commit(db, (_record("project", "projects", "Nanobot retrieval work"),))
    with db.connect() as conn:
        before = "\n".join(conn.iterdump())

    for _ in range(3):
        assert db.read_snapshot().revision == 1
        assert db.query_dynamic_memories("retrieval", 5)
        assert db.read_core_memories() == ()

    with db.connect() as conn:
        after = "\n".join(conn.iterdump())
    assert after == before


def test_queries_do_not_call_initialize(tmp_path, monkeypatch) -> None:
    db = MemoryDatabase(tmp_path)
    _commit(db, (_record("project", "projects", "Nanobot retrieval work"),))
    monkeypatch.setattr(db, "initialize", lambda: (_ for _ in ()).throw(AssertionError("write")))

    assert db.read_snapshot().revision == 1
    assert db.read_core_memories() == ()
    assert db.query_dynamic_memories("nanobot", 5)[0].memory_id == "project"


def test_initialize_preserves_existing_snapshot_and_index(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    _commit(db, (_record("project", "projects", "Nanobot project"),))
    with db.connect() as conn:
        before = list(conn.iterdump())
    db.initialize()
    with db.connect() as conn:
        assert list(conn.iterdump()) == before


@pytest.mark.parametrize(
    "record",
    [
        _record("invalid", "unknown", "fact"),
        _record("invalid", "projects", "fact", sub_class=" "),
        _record("invalid", "projects", " "),
    ],
)
def test_invalid_record_does_not_replace_committed_snapshot(tmp_path, record) -> None:
    db = MemoryDatabase(tmp_path)
    _commit(db, (_record("project", "projects", "Nanobot project"),))
    before = db.read_snapshot()
    with pytest.raises(ValueError):
        _commit(db, (record,), revision=1)
    assert db.read_snapshot() == before
    assert len(db.list_raw_events()) == 1
    assert len(db.query_dynamic_memories("Nanobot")) == 1


def test_initialize_rejects_unsupported_schema(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    with db.connect() as conn:
        conn.execute("create table canonical_memories (memory_id text primary key, text text)")
    with pytest.raises(RuntimeError, match="unsupported memory schema"):
        db.initialize()


def test_dynamic_retrieval_handles_word_variants(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    _commit(db, (_record("project", "projects", "Project produces concise answers"),))
    assert [item.memory_id for item in db.query_dynamic_memories("Answer concisely")] == ["project"]
