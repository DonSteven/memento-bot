"""SQLite storage for external web knowledge."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nanobot.utils.helpers import ensure_dir

SCHEMA_VERSION = 2
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

    def initialize(self, vector_dim: int) -> None:
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

            existing_version = int(existing_version_row["value"]) if existing_version_row else None
            existing_backend = str(existing_backend_row["value"]) if existing_backend_row else None
            existing_dim = int(existing_dim_row["value"]) if existing_dim_row else None

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

                create virtual table if not exists page_parent_fts using fts5(
                    text,
                    content='page_parents',
                    content_rowid='parent_id'
                );

                create virtual table if not exists parent_child_fts using fts5(
                    text,
                    content='parent_children',
                    content_rowid='child_id'
                );
                """
            )

            if self.vec_backend == "sqlite-vec":
                conn.execute(
                    f"create virtual table if not exists page_parent_vec using vec0(embedding float[{vector_dim}])"
                )
                conn.execute(
                    f"create virtual table if not exists parent_child_vec using vec0(embedding float[{vector_dim}])"
                )
            else:
                conn.executescript(
                    """
                    create table if not exists page_parent_vec (
                        rowid integer primary key,
                        embedding_json text not null
                    );

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
        parent_embeddings: list[list[float]],
        children: list[dict[str, Any]],
        child_embeddings: list[list[float]],
        now: str,
    ) -> dict[str, Any]:
        if len(parents) != len(parent_embeddings):
            raise ValueError("parent_embeddings must align with parents")
        if len(children) != len(child_embeddings):
            raise ValueError("child_embeddings must align with children")
        if not parents or not children:
            raise ValueError("parents and children must not be empty")

        if self._vector_dim is None:
            first_vector = parent_embeddings[0] if parent_embeddings else child_embeddings[0]
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
            for parent_index, (parent_text, parent_embedding) in enumerate(zip(parents, parent_embeddings)):
                parent_cursor = conn.execute(
                    "insert into page_parents(page_id, parent_index, text) values (?, ?, ?)",
                    (page_id, parent_index, parent_text),
                )
                parent_id = int(parent_cursor.lastrowid)
                parent_id_by_index[parent_index] = parent_id
                self._insert_parent_fts(conn, parent_id, parent_text)
                self._upsert_vector_row(conn, "page_parent_vec", parent_id, parent_embedding)

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
            conn.execute("insert into page_parent_fts(page_parent_fts) values ('rebuild')")
            conn.execute("insert into parent_child_fts(parent_child_fts) values ('rebuild')")

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

    def search_parent_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        if self.vec_backend == "sqlite-vec":
            with self.connect() as conn:
                rows = conn.execute(
                    """
                    with vec_hits as (
                        select rowid as parent_id, distance
                        from page_parent_vec
                        where embedding match ? and k = ?
                    )
                    select pp.parent_id, pp.page_id, pp.parent_index, pp.text as parent_text,
                           wp.title, wp.final_url, wp.is_partial,
                           vec_hits.distance as rank_score
                    from vec_hits
                    join page_parents pp on pp.parent_id = vec_hits.parent_id
                    join web_pages wp on wp.page_id = pp.page_id
                    order by vec_hits.distance asc, pp.parent_id asc
                    limit ?
                    """,
                    (_vector_json(query_vector), limit, limit),
                ).fetchall()
            return [dict(row) for row in rows]

        with self.connect() as conn:
            rows = conn.execute(
                """
                select pp.parent_id, pp.page_id, pp.parent_index, pp.text as parent_text,
                       wp.title, wp.final_url, wp.is_partial, pv.embedding_json
                from page_parent_vec pv
                join page_parents pp on pp.parent_id = pv.rowid
                join web_pages wp on wp.page_id = pp.page_id
                """
            ).fetchall()
        hits = []
        for row in rows:
            embedding = json.loads(str(row["embedding_json"]))
            hits.append(
                {
                    "parent_id": int(row["parent_id"]),
                    "page_id": int(row["page_id"]),
                    "parent_index": int(row["parent_index"]),
                    "parent_text": str(row["parent_text"]),
                    "title": str(row["title"]),
                    "final_url": str(row["final_url"]),
                    "is_partial": int(row["is_partial"]),
                    "rank_score": 1.0 - _cosine_similarity(query_vector, [float(value) for value in embedding]),
                }
            )
        hits.sort(key=lambda item: (float(item["rank_score"]), int(item["parent_id"])))
        return hits[:limit]

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
                       vec_hits.distance as rank_score
                from vec_hits
                join parent_children pc on pc.child_id = vec_hits.child_id
                join page_parents pp on pp.parent_id = pc.parent_id
                join web_pages wp on wp.page_id = pc.page_id
                {where_parent}
                order by vec_hits.distance asc, pc.child_id asc
                limit ?
            """
            params: list[Any] = [_vector_json(query_vector), max(limit * 5, 20), *parent_params, limit]
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
                    "rank_score": 1.0 - _cosine_similarity(query_vector, [float(value) for value in embedding]),
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
        fused: dict[Any, dict[str, Any]] = {}
        for source_name, rows in result_sets.items():
            for rank, row in enumerate(rows, start=1):
                key = row[key_field]
                current = fused.setdefault(key, {**row, "sources": [], "rrf_score": 0.0})
                current["rrf_score"] = float(current.get("rrf_score", 0.0)) + (1.0 / (_RRF_K + rank))
                current[f"{source_name}_rank"] = rank
                current.setdefault("sources", [])
                if source_name not in current["sources"]:
                    current["sources"].append(source_name)
                for item_key, item_value in row.items():
                    current.setdefault(item_key, item_value)

        ranked = sorted(
            fused.values(),
            key=lambda item: (-float(item.get("rrf_score", 0.0)), int(item[key_field])),
        )
        return ranked[:limit]

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
            self._delete_vector_row(conn, "page_parent_vec", int(row["parent_id"]))
        conn.execute("delete from page_parents where page_id = ?", (page_id,))

    @staticmethod
    def _insert_parent_fts(conn: sqlite3.Connection, parent_id: int, parent_text: str) -> None:
        conn.execute(
            "insert into page_parent_fts(rowid, text) values (?, ?)",
            (parent_id, parent_text),
        )

    @staticmethod
    def _delete_parent_fts(conn: sqlite3.Connection, parent_id: int, parent_text: str) -> None:
        conn.execute(
            "insert into page_parent_fts(page_parent_fts, rowid, text) values ('delete', ?, ?)",
            (parent_id, parent_text),
        )

    @staticmethod
    def _insert_child_fts(conn: sqlite3.Connection, child_id: int, child_text: str) -> None:
        conn.execute(
            "insert into parent_child_fts(rowid, text) values (?, ?)",
            (child_id, child_text),
        )

    @staticmethod
    def _delete_child_fts(conn: sqlite3.Connection, child_id: int, child_text: str) -> None:
        conn.execute(
            "insert into parent_child_fts(parent_child_fts, rowid, text) values ('delete', ?, ?)",
            (child_id, child_text),
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
