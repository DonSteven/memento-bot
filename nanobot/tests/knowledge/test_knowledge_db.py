# Knowledge test content:
# - verifies external web knowledge fails fast when sqlite-vec dependencies are unavailable
# - verifies page snapshots deduplicate identical content and atomically replace changed content
# - verifies parent/child external-content FTS indexes can be rebuilt from the source tables

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


def test_initialize_is_idempotent_for_same_vector_dim(tmp_path) -> None:
    db = WebKnowledgeDatabase(tmp_path, vec_backend="array")

    db.initialize(3)
    db.initialize(3)

    reopened = WebKnowledgeDatabase(tmp_path, vec_backend="array")
    reopened.initialize(3)

    assert reopened.get_schema_version() == SCHEMA_VERSION


def test_upsert_page_snapshot_deduplicates_and_replaces_changed_parent_child_content(tmp_path) -> None:
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
        parents=[
            "# Linux 6.9 Release\n\nLinux 6.9 was released in June 2024.",
            "The release included filesystem and scheduler updates.",
        ],
        parent_embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        children=[
            {"parent_index": 0, "child_index": 0, "text": "# Linux 6.9 Release\n\nLinux 6.9 was released in June 2024."},
            {"parent_index": 1, "child_index": 0, "text": "The release included filesystem and scheduler updates."},
        ],
        child_embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        now="2026-04-14T22:00:00",
    )

    assert first == {"status": "inserted", "page_id": 1}
    assert db.get_schema_version() == SCHEMA_VERSION
    assert len(db.list_pages()) == 1
    assert len(db.list_page_parents()) == 2
    assert len(db.list_parent_children()) == 2
    original_page = db.get_page_by_final_url("https://example.com/linux-6-9")
    assert original_page is not None
    original_updated_at = original_page["updated_at"]

    unchanged = db.upsert_page_snapshot(
        source_url="https://example.com/linux-6-9",
        final_url="https://example.com/linux-6-9",
        title="Linux 6.9 Release",
        extractor="readability",
        status=200,
        content_hash="hash-1",
        raw_text="Linux 6.9 was released in June 2024.",
        is_partial=False,
        parents=["unused"],
        parent_embeddings=[[1.0, 0.0, 0.0]],
        children=[{"parent_index": 0, "child_index": 0, "text": "unused"}],
        child_embeddings=[[1.0, 0.0, 0.0]],
        now="2026-04-14T22:05:00",
    )

    assert unchanged == {"status": "unchanged", "page_id": 1}
    assert len(db.list_page_parents()) == 2
    assert len(db.list_parent_children()) == 2
    unchanged_page = db.get_page_by_final_url("https://example.com/linux-6-9")
    assert unchanged_page is not None
    assert unchanged_page["updated_at"] == original_updated_at
    assert unchanged_page["last_seen_at"] == "2026-04-14T22:05:00"

    updated = db.upsert_page_snapshot(
        source_url="https://example.com/linux-6-9",
        final_url="https://example.com/linux-6-9",
        title="Linux 6.9 Release Notes",
        extractor="jina",
        status=200,
        content_hash="hash-2",
        raw_text="Linux 6.9 release notes confirmed the June 2024 release date.",
        is_partial=True,
        parents=["# Linux 6.9 Release Notes\n\nLinux 6.9 release notes confirmed the June 2024 release date."],
        parent_embeddings=[[0.8, 0.2, 0.0]],
        children=[
            {
                "parent_index": 0,
                "child_index": 0,
                "text": "# Linux 6.9 Release Notes\n\nLinux 6.9 release notes confirmed the June 2024 release date.",
            }
        ],
        child_embeddings=[[0.8, 0.2, 0.0]],
        now="2026-04-14T22:10:00",
    )

    pages = db.list_pages()
    parents = db.list_page_parents()
    children = db.list_parent_children()

    assert updated == {"status": "updated", "page_id": 1}
    assert len(pages) == 1
    assert pages[0]["title"] == "Linux 6.9 Release Notes"
    assert pages[0]["extractor"] == "jina"
    assert pages[0]["is_partial"] == 1
    assert pages[0]["content_hash"] == "hash-2"
    assert pages[0]["last_seen_at"] == "2026-04-14T22:10:00"
    assert len(parents) == 1
    assert "release notes confirmed" in parents[0]["text"]
    assert len(children) == 1
    assert "release notes confirmed" in children[0]["text"]
    assert db.search_child_fts("scheduler", 5) == []

    with db.connect() as conn:
        parent_vec_count = conn.execute("select count(*) as count from page_parent_vec").fetchone()["count"]
        child_vec_count = conn.execute("select count(*) as count from parent_child_vec").fetchone()["count"]
    assert int(parent_vec_count) == 1
    assert int(child_vec_count) == 1


def test_rrf_fuse_prefers_higher_combined_score_and_breaks_ties_by_key() -> None:
    fused = WebKnowledgeDatabase.rrf_fuse(
        {
            "fts": [
                {"parent_id": 2, "title": "Second"},
                {"parent_id": 1, "title": "First"},
            ],
            "vec": [
                {"parent_id": 1, "title": "First"},
                {"parent_id": 2, "title": "Second"},
            ],
        },
        key_field="parent_id",
        limit=5,
    )

    assert [item["parent_id"] for item in fused] == [1, 2]
    assert set(fused[0]["sources"]) == {"fts", "vec"}
    assert fused[0]["rrf_score"] >= fused[1]["rrf_score"]


def test_rebuild_indexes_restores_parent_and_child_fts_queries(tmp_path) -> None:
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
        parents=[
            "# Nanobot Web Knowledge\n\nNanobot stores fetched webpages in SQLite.",
            "The local retrieval stack combines FTS5 and vector search.",
        ],
        parent_embeddings=[[0.0, 1.0, 0.0], [0.0, 0.9, 0.1]],
        children=[
            {"parent_index": 0, "child_index": 0, "text": "# Nanobot Web Knowledge\n\nNanobot stores fetched webpages in SQLite."},
            {"parent_index": 1, "child_index": 0, "text": "The local retrieval stack combines FTS5 and vector search."},
        ],
        child_embeddings=[[0.0, 1.0, 0.0], [0.0, 0.9, 0.1]],
        now="2026-04-14T22:00:00",
    )

    assert len(db.search_parent_fts("SQLite", 5)) == 1
    assert len(db.search_child_fts("vector search", 5)) == 1

    with db.connect() as conn:
        conn.execute(
            "update page_parents set text = ? where parent_id = 2",
            ("The local retrieval stack now highlights hybrid evidence ranking.",),
        )
        conn.execute(
            "update parent_children set text = ? where child_id = 2",
            ("The child evidence now emphasizes hybrid ranking.",),
        )

    assert db.search_parent_fts("highlights", 5) == []
    assert db.search_child_fts("emphasizes", 5) == []

    db.rebuild_indexes()

    assert len(db.search_parent_fts("highlights", 5)) == 1
    assert len(db.search_child_fts("emphasizes", 5)) == 1
