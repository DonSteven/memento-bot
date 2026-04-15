"""SQLite storage for external web knowledge."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nanobot.utils.helpers import ensure_dir

SCHEMA_VERSION = 1
_RRF_K = 60


def _build_fts_query(query: str) -> str:
    """Build a tolerant FTS query from free-form text."""
    raw_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_]+", query)]
    terms: list[str] = []
    seen: set[str] = set()

    for raw in raw_terms:
        if len(raw) < 3:
            continue

        candidates = [raw]
        for suffix in ("ly", "ing", "ed", "es", "s"):
            if raw.endswith(suffix) and len(raw) - len(suffix) >= 4:
                candidates.append(raw[: -len(suffix)])

        for candidate in candidates:
            token = candidate.strip("_")
            if len(token) < 3:
                continue
            normalized = f"{token}*"
            if normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)

    return " OR ".join(terms)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot / (left_norm * right_norm)


def _vector_json(vector: list[float]) -> str:
    return json.dumps([float(value) for value in vector], ensure_ascii=False, separators=(",", ":"))


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
        sqlite_vec.load(conn)
    except AttributeError as exc:
        raise RuntimeError("The installed sqlite-vec package does not expose sqlite_vec.load().") from exc
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to load sqlite-vec extension: {exc}") from exc


class WebKnowledgeDatabase:
    """SQLite persistence for fetched web pages, summaries, and chunks."""

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

    def initialize(self, vector_dim: int) -> None:
        if vector_dim <= 0:
            raise ValueError("vector_dim must be positive")
        if self._vector_dim is None:
            self._vector_dim = vector_dim
        elif self._vector_dim != vector_dim:
            raise RuntimeError(
                f"External web knowledge already initialized with vector_dim={self._vector_dim}, "
                f"got {vector_dim}."
            )

        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists metadata (
                    key text primary key,
                    value text not null
                );

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

                create table if not exists page_summaries (
                    summary_id integer primary key autoincrement,
                    page_id integer not null unique references web_pages(page_id) on delete cascade,
                    summary_text text not null
                );

                create table if not exists page_chunks (
                    chunk_id integer primary key autoincrement,
                    page_id integer not null references web_pages(page_id) on delete cascade,
                    chunk_index integer not null,
                    text text not null,
                    unique(page_id, chunk_index)
                );

                create virtual table if not exists page_summary_fts using fts5(
                    summary_text,
                    content='page_summaries',
                    content_rowid='summary_id'
                );

                create virtual table if not exists page_chunk_fts using fts5(
                    text,
                    content='page_chunks',
                    content_rowid='chunk_id'
                );
                """
            )
            if self.vec_backend == "sqlite-vec":
                conn.execute(
                    f"create virtual table if not exists summary_vec using vec0(embedding float[{vector_dim}])"
                )
                conn.execute(
                    f"create virtual table if not exists chunk_vec using vec0(embedding float[{vector_dim}])"
                )
            else:
                conn.executescript(
                    """
                    create table if not exists summary_vec (
                        rowid integer primary key,
                        embedding_json text not null
                    );

                    create table if not exists chunk_vec (
                        rowid integer primary key,
                        embedding_json text not null
                    );
                    """
                )

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
            row = conn.execute(
                "select value from metadata where key = 'schema_version'"
            ).fetchone()
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

    def list_page_summaries(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                select summary_id, page_id, summary_text
                from page_summaries
                order by summary_id asc
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_page_chunks(self, page_id: int | None = None) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        query = (
            "select chunk_id, page_id, chunk_index, text from page_chunks "
            + ("where page_id = ? " if page_id is not None else "")
            + "order by page_id asc, chunk_index asc, chunk_id asc"
        )
        with self.connect() as conn:
            rows = conn.execute(query, (page_id,) if page_id is not None else ()).fetchall()
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
        summary_text: str,
        summary_embedding: list[float],
        chunks: list[str],
        chunk_embeddings: list[list[float]],
        now: str,
    ) -> dict[str, Any]:
        if len(chunks) != len(chunk_embeddings):
            raise ValueError("chunk_embeddings must align with chunks")
        if self._vector_dim is None:
            self.initialize(len(summary_embedding))

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

            summary_cursor = conn.execute(
                "insert into page_summaries(page_id, summary_text) values (?, ?)",
                (page_id, summary_text),
            )
            summary_id = int(summary_cursor.lastrowid)
            self._insert_summary_fts(conn, summary_id, summary_text)
            self._upsert_vector_row(conn, "summary_vec", summary_id, summary_embedding)

            for chunk_index, (chunk_text, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
                chunk_cursor = conn.execute(
                    "insert into page_chunks(page_id, chunk_index, text) values (?, ?, ?)",
                    (page_id, chunk_index, chunk_text),
                )
                chunk_id = int(chunk_cursor.lastrowid)
                self._insert_chunk_fts(conn, chunk_id, chunk_text)
                self._upsert_vector_row(conn, "chunk_vec", chunk_id, chunk_embedding)

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
            conn.execute("insert into page_summary_fts(page_summary_fts) values ('rebuild')")
            conn.execute("insert into page_chunk_fts(page_chunk_fts) values ('rebuild')")

    def search_summary_fts(self, query: str, limit: int) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        fts_query = _build_fts_query(query)
        if not fts_query:
            return []

        with self.connect() as conn:
            rows = conn.execute(
                """
                select wp.page_id, wp.title, wp.final_url, wp.is_partial,
                       ps.summary_id, ps.summary_text,
                       bm25(page_summary_fts) as rank_score
                from page_summary_fts
                join page_summaries ps on ps.summary_id = page_summary_fts.rowid
                join web_pages wp on wp.page_id = ps.page_id
                where page_summary_fts match ?
                order by bm25(page_summary_fts), ps.summary_id asc
                limit ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_summary_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        if self.vec_backend == "sqlite-vec":
            with self.connect() as conn:
                rows = conn.execute(
                    """
                    with vec_hits as (
                        select rowid as summary_id, distance
                        from summary_vec
                        where embedding match ? and k = ?
                    )
                    select wp.page_id, wp.title, wp.final_url, wp.is_partial,
                           ps.summary_id, ps.summary_text,
                           vec_hits.distance as rank_score
                    from vec_hits
                    join page_summaries ps on ps.summary_id = vec_hits.summary_id
                    join web_pages wp on wp.page_id = ps.page_id
                    order by vec_hits.distance asc, ps.summary_id asc
                    limit ?
                    """,
                    (_vector_json(query_vector), limit, limit),
                ).fetchall()
            return [dict(row) for row in rows]

        with self.connect() as conn:
            rows = conn.execute(
                """
                select wp.page_id, wp.title, wp.final_url, wp.is_partial,
                       ps.summary_id, ps.summary_text, sv.embedding_json
                from summary_vec sv
                join page_summaries ps on ps.summary_id = sv.rowid
                join web_pages wp on wp.page_id = ps.page_id
                """
            ).fetchall()
        hits = []
        for row in rows:
            embedding = json.loads(str(row["embedding_json"]))
            hits.append({
                "page_id": int(row["page_id"]),
                "title": str(row["title"]),
                "final_url": str(row["final_url"]),
                "is_partial": int(row["is_partial"]),
                "summary_id": int(row["summary_id"]),
                "summary_text": str(row["summary_text"]),
                "rank_score": 1.0 - _cosine_similarity(query_vector, [float(value) for value in embedding]),
            })
        hits.sort(key=lambda item: (float(item["rank_score"]), int(item["summary_id"])))
        return hits[:limit]

    def search_chunk_fts(self, query: str, page_ids: list[int], limit: int) -> list[dict[str, Any]]:
        if not self.db_path.exists() or not page_ids:
            return []
        fts_query = _build_fts_query(query)
        if not fts_query:
            return []

        placeholders = ", ".join("?" for _ in page_ids)
        sql = f"""
            select c.chunk_id, c.page_id, c.chunk_index, c.text,
                   wp.title, wp.final_url, wp.is_partial,
                   bm25(page_chunk_fts) as rank_score
            from page_chunk_fts
            join page_chunks c on c.chunk_id = page_chunk_fts.rowid
            join web_pages wp on wp.page_id = c.page_id
            where page_chunk_fts match ? and c.page_id in ({placeholders})
            order by bm25(page_chunk_fts), c.chunk_id asc
            limit ?
        """
        params: list[Any] = [fts_query, *page_ids, limit]
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def search_chunk_vector(
        self,
        query_vector: list[float],
        page_ids: list[int],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not self.db_path.exists() or not page_ids:
            return []
        placeholders = ", ".join("?" for _ in page_ids)
        if self.vec_backend == "sqlite-vec":
            sql = f"""
                with vec_hits as (
                    select rowid as chunk_id, distance
                    from chunk_vec
                    where embedding match ? and k = ?
                )
                select c.chunk_id, c.page_id, c.chunk_index, c.text,
                       wp.title, wp.final_url, wp.is_partial,
                       vec_hits.distance as rank_score
                from vec_hits
                join page_chunks c on c.chunk_id = vec_hits.chunk_id
                join web_pages wp on wp.page_id = c.page_id
                where c.page_id in ({placeholders})
                order by vec_hits.distance asc, c.chunk_id asc
                limit ?
            """
            params: list[Any] = [_vector_json(query_vector), max(limit * 5, 20), *page_ids, limit]
            with self.connect() as conn:
                rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

        sql = f"""
            select c.chunk_id, c.page_id, c.chunk_index, c.text,
                   wp.title, wp.final_url, wp.is_partial,
                   cv.embedding_json
            from chunk_vec cv
            join page_chunks c on c.chunk_id = cv.rowid
            join web_pages wp on wp.page_id = c.page_id
            where c.page_id in ({placeholders})
        """
        with self.connect() as conn:
            rows = conn.execute(sql, page_ids).fetchall()
        hits = []
        for row in rows:
            embedding = json.loads(str(row["embedding_json"]))
            hits.append({
                "chunk_id": int(row["chunk_id"]),
                "page_id": int(row["page_id"]),
                "chunk_index": int(row["chunk_index"]),
                "text": str(row["text"]),
                "title": str(row["title"]),
                "final_url": str(row["final_url"]),
                "is_partial": int(row["is_partial"]),
                "rank_score": 1.0 - _cosine_similarity(query_vector, [float(value) for value in embedding]),
            })
        hits.sort(key=lambda item: (float(item["rank_score"]), int(item["chunk_id"])))
        return hits[:limit]

    @staticmethod
    def rrf_fuse(
        result_sets: dict[str, list[dict[str, Any]]],
        *,
        key_field: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        fused: dict[Any, dict[str, Any]] = {}
        for source_name, rows in result_sets.items():
            for rank, row in enumerate(rows, start=1):
                key = row[key_field]
                current = fused.setdefault(key, {**row, "sources": []})
                current.setdefault("_rrf_score", 0.0)
                current.setdefault("sources", [])
                current["_rrf_score"] += 1.0 / (_RRF_K + rank)
                current[f"{source_name}_rank"] = rank
                if source_name not in current["sources"]:
                    current["sources"].append(source_name)
                for item_key, item_value in row.items():
                    current.setdefault(item_key, item_value)

        ranked = sorted(
            fused.values(),
            key=lambda item: (-float(item.get("_rrf_score", 0.0)), int(item[key_field])),
        )
        for item in ranked:
            item.pop("_rrf_score", None)
        return ranked[:limit]

    def _delete_page_children(self, conn: sqlite3.Connection, page_id: int) -> None:
        summary_rows = conn.execute(
            "select summary_id, summary_text from page_summaries where page_id = ?",
            (page_id,),
        ).fetchall()
        for row in summary_rows:
            self._delete_summary_fts(conn, int(row["summary_id"]), str(row["summary_text"]))
            self._delete_vector_row(conn, "summary_vec", int(row["summary_id"]))
        conn.execute("delete from page_summaries where page_id = ?", (page_id,))

        chunk_rows = conn.execute(
            "select chunk_id, text from page_chunks where page_id = ?",
            (page_id,),
        ).fetchall()
        for row in chunk_rows:
            self._delete_chunk_fts(conn, int(row["chunk_id"]), str(row["text"]))
            self._delete_vector_row(conn, "chunk_vec", int(row["chunk_id"]))
        conn.execute("delete from page_chunks where page_id = ?", (page_id,))

    @staticmethod
    def _insert_summary_fts(conn: sqlite3.Connection, summary_id: int, summary_text: str) -> None:
        conn.execute(
            "insert into page_summary_fts(rowid, summary_text) values (?, ?)",
            (summary_id, summary_text),
        )

    @staticmethod
    def _delete_summary_fts(conn: sqlite3.Connection, summary_id: int, summary_text: str) -> None:
        conn.execute(
            "insert into page_summary_fts(page_summary_fts, rowid, summary_text) values ('delete', ?, ?)",
            (summary_id, summary_text),
        )

    @staticmethod
    def _insert_chunk_fts(conn: sqlite3.Connection, chunk_id: int, chunk_text: str) -> None:
        conn.execute(
            "insert into page_chunk_fts(rowid, text) values (?, ?)",
            (chunk_id, chunk_text),
        )

    @staticmethod
    def _delete_chunk_fts(conn: sqlite3.Connection, chunk_id: int, chunk_text: str) -> None:
        conn.execute(
            "insert into page_chunk_fts(page_chunk_fts, rowid, text) values ('delete', ?, ?)",
            (chunk_id, chunk_text),
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
