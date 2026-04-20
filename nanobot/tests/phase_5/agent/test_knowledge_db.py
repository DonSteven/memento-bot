# Phase 5 test content:
# - verifies external web knowledge fails fast when sqlite-vec dependencies are unavailable
# - verifies page snapshots deduplicate identical content and atomically replace changed content
# - verifies external-content FTS indexes can be rebuilt from the source tables
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_5/agent/test_knowledge_db.py

from __future__ import annotations

import builtins
import sqlite3
import sys

import pytest

from nanobot.agent.knowledge_db import SCHEMA_VERSION, WebKnowledgeDatabase, _default_vec_loader


def test_default_vec_loader_requires_web_knowledge_dependencies(monkeypatch) -> None:
    original_import = builtins.__import__

    def _missing_sqlite_vec(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sqlite_vec":
            raise ImportError("missing sqlite_vec")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "sqlite_vec", raising=False)
    monkeypatch.setattr(builtins, "__import__", _missing_sqlite_vec)

    with pytest.raises(RuntimeError, match="uv sync --extra web_knowledge"):
        _default_vec_loader(sqlite3.connect(":memory:"))


def test_upsert_page_snapshot_deduplicates_and_replaces_changed_content(tmp_path) -> None:
    db = WebKnowledgeDatabase(tmp_path, vec_backend="array")
    db.initialize(3)

    first = db.upsert_page_snapshot(
        source_url="https://example.com/linux-6-9",
        final_url="https://example.com/linux-6-9",
        title="Linux 6.9 Release",
        extractor="readability",
        status=200,
        content_hash="hash-1",
        raw_text="Linux 6.9 was released in June 2024.",
        is_partial=False,
        summary_text="Linux 6.9 release summary.",
        summary_embedding=[1.0, 0.0, 0.0],
        chunks=[
            "# Linux 6.9 Release\n\nLinux 6.9 was released in June 2024.",
            "The release included filesystem and scheduler updates.",
        ],
        chunk_embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        now="2026-04-14T22:00:00",
    )
    assert first["status"] == "inserted"
    assert db.get_schema_version() == SCHEMA_VERSION
    assert len(db.list_pages()) == 1
    assert len(db.list_page_summaries()) == 1
    assert len(db.list_page_chunks()) == 2

    unchanged = db.upsert_page_snapshot(
        source_url="https://example.com/linux-6-9",
        final_url="https://example.com/linux-6-9",
        title="Linux 6.9 Release",
        extractor="readability",
        status=200,
        content_hash="hash-1",
        raw_text="Linux 6.9 was released in June 2024.",
        is_partial=False,
        summary_text="unused because hash is unchanged",
        summary_embedding=[1.0, 0.0, 0.0],
        chunks=["unused"],
        chunk_embeddings=[[1.0, 0.0, 0.0]],
        now="2026-04-14T22:05:00",
    )
    assert unchanged["status"] == "unchanged"
    assert len(db.list_page_summaries()) == 1
    assert len(db.list_page_chunks()) == 2

    updated = db.upsert_page_snapshot(
        source_url="https://example.com/linux-6-9",
        final_url="https://example.com/linux-6-9",
        title="Linux 6.9 Release Notes",
        extractor="jina",
        status=200,
        content_hash="hash-2",
        raw_text="Linux 6.9 release notes confirmed the June 2024 release date.",
        is_partial=True,
        summary_text="Linux 6.9 release notes summary.",
        summary_embedding=[0.8, 0.2, 0.0],
        chunks=[
            "# Linux 6.9 Release Notes\n\nLinux 6.9 release notes confirmed the June 2024 release date.",
        ],
        chunk_embeddings=[[0.8, 0.2, 0.0]],
        now="2026-04-14T22:10:00",
    )

    pages = db.list_pages()
    summaries = db.list_page_summaries()
    chunks = db.list_page_chunks()

    assert updated["status"] == "updated"
    assert updated["page_id"] == first["page_id"]
    assert len(pages) == 1
    assert pages[0]["title"] == "Linux 6.9 Release Notes"
    assert pages[0]["extractor"] == "jina"
    assert pages[0]["is_partial"] == 1
    assert pages[0]["content_hash"] == "hash-2"
    assert len(summaries) == 1
    assert summaries[0]["summary_text"] == "Linux 6.9 release notes summary."
    assert len(chunks) == 1
    assert "release notes confirmed" in chunks[0]["text"]


def test_rebuild_indexes_restores_summary_and_chunk_fts_queries(tmp_path) -> None:
    db = WebKnowledgeDatabase(tmp_path, vec_backend="array")
    db.initialize(3)
    db.upsert_page_snapshot(
        source_url="https://example.com/nanobot-kb",
        final_url="https://example.com/nanobot-kb",
        title="Nanobot Web Knowledge",
        extractor="readability",
        status=200,
        content_hash="hash-1",
        raw_text="Nanobot stores fetched webpages in SQLite and searches them locally.",
        is_partial=False,
        summary_text="Nanobot web knowledge uses SQLite for local retrieval.",
        summary_embedding=[0.0, 1.0, 0.0],
        chunks=[
            "# Nanobot Web Knowledge\n\nNanobot stores fetched webpages in SQLite.",
            "The local retrieval stack combines FTS5 and vector search.",
        ],
        chunk_embeddings=[[0.0, 1.0, 0.0], [0.0, 0.9, 0.1]],
        now="2026-04-14T22:00:00",
    )

    assert len(db.search_summary_fts("SQLite retrieval", 5)) == 1
    assert len(db.search_chunk_fts("vector search", [1], 5)) == 1

    with db.connect() as conn:
        conn.execute(
            "update page_summaries set summary_text = ? where summary_id = 1",
            ("Nanobot local retrieval now highlights SQLite evidence.",),
        )
        conn.execute(
            "update page_chunks set text = ? where chunk_id = 2",
            ("The local retrieval stack now emphasizes hybrid evidence ranking.",),
        )

    assert db.search_summary_fts("highlights", 5) == []
    assert db.search_chunk_fts("emphasizes", [1], 5) == []

    db.rebuild_indexes()

    assert len(db.search_summary_fts("highlights", 5)) == 1
    assert len(db.search_chunk_fts("emphasizes", [1], 5)) == 1


def test_default_vec_loader_enables_and_disables_extension_loading(monkeypatch) -> None:
    events: list[tuple[str, bool] | tuple[str, str]] = []

    class _FakeConnection:
        def enable_load_extension(self, enabled: bool) -> None:
            events.append(("enable", enabled))

    class _FakeSqliteVec:
        @staticmethod
        def load(conn) -> None:
            assert isinstance(conn, _FakeConnection)
            events.append(("load", "sqlite-vec"))

    monkeypatch.setitem(sys.modules, "sqlite_vec", _FakeSqliteVec)

    _default_vec_loader(_FakeConnection())

    assert events == [
        ("enable", True),
        ("load", "sqlite-vec"),
        ("enable", False),
    ]


def test_default_vec_loader_surfaces_enable_load_extension_failures(monkeypatch) -> None:
    class _FakeConnection:
        def enable_load_extension(self, enabled: bool) -> None:
            raise sqlite3.OperationalError("not authorized")

    class _FakeSqliteVec:
        @staticmethod
        def load(conn) -> None:
            raise AssertionError("sqlite_vec.load() should not be called when enabling fails")

    monkeypatch.setitem(sys.modules, "sqlite_vec", _FakeSqliteVec)

    with pytest.raises(RuntimeError, match="could not enable SQLite extension loading"):
        _default_vec_loader(_FakeConnection())


def test_default_vec_loader_disables_extension_loading_after_load_failure(monkeypatch) -> None:
    events: list[tuple[str, bool]] = []

    class _FakeConnection:
        def enable_load_extension(self, enabled: bool) -> None:
            events.append(("enable", enabled))

    class _FakeSqliteVec:
        @staticmethod
        def load(conn) -> None:
            raise sqlite3.OperationalError("boom")

    monkeypatch.setitem(sys.modules, "sqlite_vec", _FakeSqliteVec)

    with pytest.raises(RuntimeError, match="Failed to load sqlite-vec extension: boom"):
        _default_vec_loader(_FakeConnection())

    assert events == [("enable", True), ("enable", False)]
