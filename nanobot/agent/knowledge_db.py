"""SQLite storage for external web knowledge."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nanobot.agent.retrieval import (
    FTS_TOKENIZER_VERSION,
    build_fts_query,
    cosine_similarity,
    fts_index_text,
    reciprocal_rank_fuse,
    vector_json,
)
from nanobot.utils.helpers import ensure_dir

SCHEMA_VERSION = 4
_build_fts_query = build_fts_query
_cosine_similarity = cosine_similarity
_vector_json = vector_json


def _default_vec_loader(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec into the given connection."""
    try:
        import sqlite_vec
    except ImportError as exc:
        raise RuntimeError(
            "External web knowledge requires optional dependencies. "
            "Install them with `uv sync --extra web_knowledge` or equivalent."
        ) from exc

    try:
        conn.enable_load_extension(True)
    except AttributeError as exc:
        raise RuntimeError(
            "The active sqlite3 build does not expose enable_load_extension(), "
            "which is required to load sqlite-vec."
        ) from exc
    except sqlite3.Error as exc:
        raise RuntimeError(
            "The active sqlite3 build could not enable SQLite extension loading, "
            f"which is required to load sqlite-vec: {exc}"
        ) from exc
    try:
        sqlite_vec.load(conn)
    except AttributeError as exc:
        raise RuntimeError("The installed sqlite-vec package does not expose sqlite_vec.load().") from exc
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to load sqlite-vec extension: {exc}") from exc
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass


class WebKnowledgeDatabase:
    """SQLite persistence for fetched web pages, parent blocks, and child blocks."""

    def __init__(
        self,
        workspace: Path,
        *,
        storage_dir: Path | None = None,
        vec_backend: str = "sqlite-vec",
        vec_loader: Callable[[sqlite3.Connection], None] | None = None,
    ):
        if vec_backend not in {"sqlite-vec", "array"}:
            raise ValueError(f"Unsupported vector backend: {vec_backend}")

        self.knowledge_dir = ensure_dir(workspace / "knowledge")
        self.storage_dir = ensure_dir(storage_dir or self.knowledge_dir)
        self.db_path = self.storage_dir / "web_knowledge.db"
        self.vec_backend = vec_backend
        self._vec_loader = vec_loader or (_default_vec_loader if vec_backend == "sqlite-vec" else None)
        self._vector_dim: int | None = None

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if self.vec_backend == "sqlite-vec" and self._vec_loader is not None:
            self._vec_loader(conn)
        return conn

    def initialize(
        self,
        vector_dim: int,
        *,
        embedding_provider: str = "dashscope",
        embedding_model: str = "text-embedding-v4",
    ) -> None:
        if vector_dim <= 0:
            raise ValueError("vector_dim must be positive")
        if self._vector_dim is not None and self._vector_dim != vector_dim:
            raise RuntimeError(
                f"External web knowledge already initialized with vector_dim={self._vector_dim}, "
                f"got {vector_dim}."
            )

        with self.connect() as conn:
            conn.execute(
                """
                create table if not exists metadata (
                    key text primary key,
                    value text not null
                )
                """
            )
            existing_version_row = conn.execute(
                "select value from metadata where key = 'schema_version'"
            ).fetchone()
            existing_backend_row = conn.execute(
                "select value from metadata where key = 'vec_backend'"
            ).fetchone()
            existing_dim_row = conn.execute(
                "select value from metadata where key = 'vector_dim'"
            ).fetchone()
            existing_tokenizer_row = conn.execute(
                "select value from metadata where key = 'fts_tokenizer_version'"
            ).fetchone()

            existing_version = int(existing_version_row["value"]) if existing_version_row else None
            existing_backend = str(existing_backend_row["value"]) if existing_backend_row else None
            existing_dim = int(existing_dim_row["value"]) if existing_dim_row else None
            existing_tokenizer = (
                str(existing_tokenizer_row["value"]) if existing_tokenizer_row else None
            )
            for key, expected in {
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
            }.items():
                row = conn.execute("select value from metadata where key=?", (key,)).fetchone()
                if row is not None and str(row["value"]) != expected:
                    raise RuntimeError(
                        f"External web knowledge index metadata mismatch for {key}; rebuild the knowledge indexes."
                    )

            if existing_version is not None and existing_version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"External web knowledge database schema_version={existing_version} is incompatible with "
                    f"the current parent-child schema_version={SCHEMA_VERSION}. Recreate the database."
                )
            if existing_backend is not None and existing_backend != self.vec_backend:
                raise RuntimeError(
                    f"External web knowledge already initialized with vec_backend={existing_backend}, "
                    f"got {self.vec_backend}."
                )
            if existing_dim is not None and existing_dim != vector_dim:
                raise RuntimeError(
                    f"External web knowledge already initialized with vector_dim={existing_dim}, "
                    f"got {vector_dim}."
                )
            if existing_tokenizer is not None and existing_tokenizer != FTS_TOKENIZER_VERSION:
                raise RuntimeError(
                    "External web knowledge FTS tokenizer changed; rebuild the knowledge indexes."
                )

            conn.executescript(
                """
                create table if not exists web_pages (
                    page_id integer primary key autoincrement,
                    source_url text not null,
                    final_url text not null unique,
                    title text not null default '',
                    extractor text not null default '',
                    status integer not null default 0,
                    content_hash text not null,
                    raw_text text not null,
                    is_partial integer not null default 0,
                    created_at text not null,
                    updated_at text not null,
                    last_seen_at text not null
                );

                create table if not exists page_parents (
                    parent_id integer primary key autoincrement,
                    page_id integer not null references web_pages(page_id) on delete cascade,
                    parent_index integer not null,
                    text text not null,
                    unique(page_id, parent_index)
                );

                create table if not exists parent_children (
                    child_id integer primary key autoincrement,
                    parent_id integer not null references page_parents(parent_id) on delete cascade,
                    page_id integer not null references web_pages(page_id) on delete cascade,
                    child_index integer not null,
                    text text not null,
                    unique(parent_id, child_index)
                );

                create virtual table if not exists page_parent_fts using fts5(tokens);
                create virtual table if not exists parent_child_fts using fts5(tokens);
                """
            )

            if self.vec_backend == "sqlite-vec":
                conn.execute(
                    f"create virtual table if not exists parent_child_vec using vec0(embedding float[{vector_dim}])"
                )
            else:
                conn.executescript(
                    """
                    create table if not exists parent_child_vec (
                        rowid integer primary key,
                        embedding_json text not null
                    );
                    """
                )

            self._vector_dim = existing_dim or vector_dim
            conn.execute(
                """
                insert into metadata(key, value) values('schema_version', ?)
                on conflict(key) do update set value=excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            conn.execute(
                """
                insert into metadata(key, value) values('vector_dim', ?)
                on conflict(key) do update set value=excluded.value
                """,
                (str(vector_dim),),
            )
            conn.execute(
                """
                insert into metadata(key, value) values('vec_backend', ?)
                on conflict(key) do update set value=excluded.value
                """,
                (self.vec_backend,),
            )
            conn.execute(
                "insert into metadata(key, value) values('fts_tokenizer_version', ?) "
                "on conflict(key) do update set value=excluded.value",
                (FTS_TOKENIZER_VERSION,),
            )
            for key, value in {
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
            }.items():
                conn.execute(
                    "insert or ignore into metadata(key, value) values(?, ?)", (key, value)
                )

    def has_pages(self) -> bool:
        if not self.db_path.exists():
            return False
        with self.connect() as conn:
            row = conn.execute("select count(*) as count from web_pages").fetchone()
        return bool(row and int(row["count"]) > 0)

    def get_schema_version(self) -> int:
        if not self.db_path.exists():
            return 0
        with self.connect() as conn:
            row = conn.execute("select value from metadata where key = 'schema_version'").fetchone()
        return int(row["value"]) if row else 0

    def get_page_by_final_url(self, final_url: str) -> dict[str, Any] | None:
        if not self.db_path.exists():
            return None
        with self.connect() as conn:
            row = conn.execute(
                """
                select page_id, source_url, final_url, title, extractor, status,
                       content_hash, raw_text, is_partial, created_at, updated_at, last_seen_at
                from web_pages
                where final_url = ?
                """,
                (final_url,),
            ).fetchone()
        return dict(row) if row else None

    def list_pages(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                select page_id, source_url, final_url, title, extractor, status,
                       content_hash, raw_text, is_partial, created_at, updated_at, last_seen_at
                from web_pages
                order by page_id asc
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_page_parents(self, page_id: int | None = None) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        query = (
            "select parent_id, page_id, parent_index, text from page_parents "
            + ("where page_id = ? " if page_id is not None else "")
            + "order by page_id asc, parent_index asc, parent_id asc"
        )
        with self.connect() as conn:
            rows = conn.execute(query, (page_id,) if page_id is not None else ()).fetchall()
        return [dict(row) for row in rows]

    def list_parent_children(
        self,
        page_id: int | None = None,
        parent_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if page_id is not None:
            clauses.append("page_id = ?")
            params.append(page_id)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            params.append(parent_id)
        where_sql = f"where {' and '.join(clauses)} " if clauses else ""
        query = (
            "select child_id, parent_id, page_id, child_index, text from parent_children "
            + where_sql
            + "order by page_id asc, parent_id asc, child_index asc, child_id asc"
        )
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def upsert_page_snapshot(
        self,
        *,
        source_url: str,
        final_url: str,
        title: str,
        extractor: str,
        status: int,
        content_hash: str,
        raw_text: str,
        is_partial: bool,
        parents: list[str],
        children: list[dict[str, Any]],
        child_embeddings: list[list[float]],
        now: str,
    ) -> dict[str, Any]:
        if len(children) != len(child_embeddings):
            raise ValueError("child_embeddings must align with children")
        if not parents or not children:
            raise ValueError("parents and children must not be empty")

        if self._vector_dim is None:
            first_vector = child_embeddings[0]
            self.initialize(len(first_vector))

        with self.connect() as conn:
            existing = conn.execute(
                "select page_id, content_hash from web_pages where final_url = ?",
                (final_url,),
            ).fetchone()
            if existing and str(existing["content_hash"]) == content_hash:
                conn.execute(
                    """
                    update web_pages
                    set source_url = ?,
                        title = ?,
                        extractor = ?,
                        status = ?,
                        is_partial = ?,
                        last_seen_at = ?
                    where page_id = ?
                    """,
                    (
                        source_url,
                        title,
                        extractor,
                        status,
                        int(is_partial),
                        now,
                        int(existing["page_id"]),
                    ),
                )
                return {"status": "unchanged", "page_id": int(existing["page_id"])}

            if existing:
                page_id = int(existing["page_id"])
                conn.execute(
                    """
                    update web_pages
                    set source_url = ?,
                        title = ?,
                        extractor = ?,
                        status = ?,
                        content_hash = ?,
                        raw_text = ?,
                        is_partial = ?,
                        updated_at = ?,
                        last_seen_at = ?
                    where page_id = ?
                    """,
                    (
                        source_url,
                        title,
                        extractor,
                        status,
                        content_hash,
                        raw_text,
                        int(is_partial),
                        now,
                        now,
                        page_id,
                    ),
                )
                self._delete_page_children(conn, page_id)
                write_status = "updated"
            else:
                cursor = conn.execute(
                    """
                    insert into web_pages(
                        source_url, final_url, title, extractor, status, content_hash,
                        raw_text, is_partial, created_at, updated_at, last_seen_at
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_url,
                        final_url,
                        title,
                        extractor,
                        status,
                        content_hash,
                        raw_text,
                        int(is_partial),
                        now,
                        now,
                        now,
                    ),
                )
                page_id = int(cursor.lastrowid)
                write_status = "inserted"

            parent_id_by_index: dict[int, int] = {}
            for parent_index, parent_text in enumerate(parents):
                parent_cursor = conn.execute(
                    "insert into page_parents(page_id, parent_index, text) values (?, ?, ?)",
                    (page_id, parent_index, parent_text),
                )
                parent_id = int(parent_cursor.lastrowid)
                parent_id_by_index[parent_index] = parent_id
                self._insert_parent_fts(conn, parent_id, parent_text)

            for child_record, child_embedding in zip(children, child_embeddings):
                parent_index = int(child_record["parent_index"])
                child_index = int(child_record["child_index"])
                child_text = str(child_record["text"])
                parent_id = parent_id_by_index[parent_index]
                child_cursor = conn.execute(
                    """
                    insert into parent_children(parent_id, page_id, child_index, text)
                    values (?, ?, ?, ?)
                    """,
                    (parent_id, page_id, child_index, child_text),
                )
                child_id = int(child_cursor.lastrowid)
                self._insert_child_fts(conn, child_id, child_text)
                self._upsert_vector_row(conn, "parent_child_vec", child_id, child_embedding)

            return {"status": write_status, "page_id": page_id}

    def mark_page_seen(
        self,
        *,
        final_url: str,
        source_url: str,
        title: str,
        extractor: str,
        status: int,
        is_partial: bool,
        seen_at: str,
    ) -> dict[str, Any]:
        if not self.db_path.exists():
            raise RuntimeError("External web knowledge database is not initialized")
        with self.connect() as conn:
            row = conn.execute(
                "select page_id from web_pages where final_url = ?",
                (final_url,),
            ).fetchone()
            if not row:
                raise RuntimeError(f"Cannot mark unseen page as seen: {final_url}")
            page_id = int(row["page_id"])
            conn.execute(
                """
                update web_pages
                set source_url = ?,
                    title = ?,
                    extractor = ?,
                    status = ?,
                    is_partial = ?,
                    last_seen_at = ?
                where page_id = ?
                """,
                (
                    source_url,
                    title,
                    extractor,
                    status,
                    int(is_partial),
                    seen_at,
                    page_id,
                ),
            )
        return {"status": "unchanged", "page_id": page_id}

    def rebuild_indexes(self) -> None:
        if not self.db_path.exists():
            return
        with self.connect() as conn:
            conn.execute("delete from page_parent_fts")
            conn.execute("delete from parent_child_fts")
            for row in conn.execute("select parent_id, text from page_parents"):
                self._insert_parent_fts(conn, int(row["parent_id"]), str(row["text"]))
            for row in conn.execute("select child_id, text from parent_children"):
                self._insert_child_fts(conn, int(row["child_id"]), str(row["text"]))

    def search_parent_fts(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        fts_query = _build_fts_query(query)
        if not fts_query:
            return []

        with self.connect() as conn:
            rows = conn.execute(
                """
                select pp.parent_id, pp.page_id, pp.parent_index, pp.text as parent_text,
                       wp.title, wp.final_url, wp.is_partial,
                       bm25(page_parent_fts) as rank_score
                from page_parent_fts
                join page_parents pp on pp.parent_id = page_parent_fts.rowid
                join web_pages wp on wp.page_id = pp.page_id
                where page_parent_fts match ?
                order by bm25(page_parent_fts), pp.parent_id asc
                limit ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_child_fts(
        self,
        query: str,
        limit: int,
        *,
        parent_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        fts_query = _build_fts_query(query)
        if not fts_query:
            return []

        where_parent = ""
        params: list[Any] = [fts_query]
        if parent_ids:
            placeholders = ", ".join("?" for _ in parent_ids)
            where_parent = f" and pc.parent_id in ({placeholders})"
            params.extend(parent_ids)
        params.append(limit)

        sql = f"""
            select pc.child_id, pc.parent_id, pc.page_id, pc.child_index, pc.text,
                   pp.parent_index, pp.text as parent_text,
                   wp.title, wp.final_url, wp.is_partial,
                   bm25(parent_child_fts) as rank_score
            from parent_child_fts
            join parent_children pc on pc.child_id = parent_child_fts.rowid
            join page_parents pp on pp.parent_id = pc.parent_id
            join web_pages wp on wp.page_id = pc.page_id
            where parent_child_fts match ?{where_parent}
            order by bm25(parent_child_fts), pc.child_id asc
            limit ?
        """
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def search_child_vector(
        self,
        query_vector: list[float],
        limit: int,
        *,
        parent_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []

        where_parent = ""
        parent_params: list[Any] = []
        if parent_ids:
            placeholders = ", ".join("?" for _ in parent_ids)
            where_parent = f" where pc.parent_id in ({placeholders})"
            parent_params.extend(parent_ids)

        if self.vec_backend == "sqlite-vec":
            sql = f"""
                with vec_hits as (
                    select rowid as child_id, distance
                    from parent_child_vec
                    where embedding match ? and k = ?
                )
                select pc.child_id, pc.parent_id, pc.page_id, pc.child_index, pc.text,
                       pp.parent_index, pp.text as parent_text,
                       wp.title, wp.final_url, wp.is_partial,
                       vec_hits.distance as rank_score,
                       1.0 - vec_distance_cosine(v.embedding, ?) as vector_similarity
                from vec_hits
                join parent_child_vec v on v.rowid = vec_hits.child_id
                join parent_children pc on pc.child_id = vec_hits.child_id
                join page_parents pp on pp.parent_id = pc.parent_id
                join web_pages wp on wp.page_id = pc.page_id
                {where_parent}
                order by vec_hits.distance asc, pc.child_id asc
                limit ?
            """
            params: list[Any] = [
                _vector_json(query_vector),
                max(limit * 5, 20),
                _vector_json(query_vector),
                *parent_params,
                limit,
            ]
            with self.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

        sql = f"""
            select pc.child_id, pc.parent_id, pc.page_id, pc.child_index, pc.text,
                   pp.parent_index, pp.text as parent_text,
                   wp.title, wp.final_url, wp.is_partial,
                   cv.embedding_json
            from parent_child_vec cv
            join parent_children pc on pc.child_id = cv.rowid
            join page_parents pp on pp.parent_id = pc.parent_id
            join web_pages wp on wp.page_id = pc.page_id
            {where_parent}
        """
        with self.connect() as conn:
            rows = conn.execute(sql, parent_params).fetchall()
        hits = []
        for row in rows:
            embedding = json.loads(str(row["embedding_json"]))
            similarity = _cosine_similarity(query_vector, [float(value) for value in embedding])
            hits.append(
                {
                    "child_id": int(row["child_id"]),
                    "parent_id": int(row["parent_id"]),
                    "page_id": int(row["page_id"]),
                    "child_index": int(row["child_index"]),
                    "text": str(row["text"]),
                    "parent_index": int(row["parent_index"]),
                    "parent_text": str(row["parent_text"]),
                    "title": str(row["title"]),
                    "final_url": str(row["final_url"]),
                    "is_partial": int(row["is_partial"]),
                    "rank_score": 1.0 - similarity,
                    "vector_similarity": similarity,
                }
            )
        hits.sort(key=lambda item: (float(item["rank_score"]), int(item["child_id"])))
        return hits[:limit]

    @staticmethod
    def rrf_fuse(
        result_sets: dict[str, list[dict[str, Any]]],
        *,
        key_field: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return reciprocal_rank_fuse(result_sets, key_field=key_field, limit=limit)

    def _delete_page_children(self, conn: sqlite3.Connection, page_id: int) -> None:
        child_rows = conn.execute(
            "select child_id, text from parent_children where page_id = ?",
            (page_id,),
        ).fetchall()
        for row in child_rows:
            self._delete_child_fts(conn, int(row["child_id"]), str(row["text"]))
            self._delete_vector_row(conn, "parent_child_vec", int(row["child_id"]))
        conn.execute("delete from parent_children where page_id = ?", (page_id,))

        parent_rows = conn.execute(
            "select parent_id, text from page_parents where page_id = ?",
            (page_id,),
        ).fetchall()
        for row in parent_rows:
            self._delete_parent_fts(conn, int(row["parent_id"]), str(row["text"]))
        conn.execute("delete from page_parents where page_id = ?", (page_id,))

    @staticmethod
    def _insert_parent_fts(conn: sqlite3.Connection, parent_id: int, parent_text: str) -> None:
        conn.execute(
            "insert into page_parent_fts(rowid, tokens) values (?, ?)",
            (parent_id, fts_index_text(parent_text)),
        )

    @staticmethod
    def _delete_parent_fts(conn: sqlite3.Connection, parent_id: int, parent_text: str) -> None:
        conn.execute(
            "delete from page_parent_fts where rowid = ?",
            (parent_id,),
        )

    @staticmethod
    def _insert_child_fts(conn: sqlite3.Connection, child_id: int, child_text: str) -> None:
        conn.execute(
            "insert into parent_child_fts(rowid, tokens) values (?, ?)",
            (child_id, fts_index_text(child_text)),
        )

    @staticmethod
    def _delete_child_fts(conn: sqlite3.Connection, child_id: int, child_text: str) -> None:
        conn.execute(
            "delete from parent_child_fts where rowid = ?",
            (child_id,),
        )

    def _upsert_vector_row(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        row_id: int,
        embedding: list[float],
    ) -> None:
        payload = _vector_json(embedding)
        if self.vec_backend == "sqlite-vec":
            conn.execute(f"delete from {table_name} where rowid = ?", (row_id,))
            conn.execute(f"insert into {table_name}(rowid, embedding) values (?, ?)", (row_id, payload))
            return

        conn.execute(
            f"""
            insert into {table_name}(rowid, embedding_json) values (?, ?)
            on conflict(rowid) do update set embedding_json=excluded.embedding_json
            """,
            (row_id, payload),
        )

    @staticmethod
    def _delete_vector_row(conn: sqlite3.Connection, table_name: str, row_id: int) -> None:
        conn.execute(f"delete from {table_name} where rowid = ?", (row_id,))
