"""SQLite persistence and data contracts for structured personal memory."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

from nanobot.agent.retrieval import (
    FTS_TOKENIZER_VERSION,
    build_fts_query,
    cosine_similarity,
    fts_index_text,
    load_sqlite_vec,
    reciprocal_rank_fuse,
    vector_json,
)
from nanobot.utils.helpers import ensure_dir

SCHEMA_VERSION = 8
CORE_MEMORY_CLASSES = ("personal_profile", "preferences", "constraints")
DYNAMIC_MEMORY_CLASSES = ("projects", "daily_life", "plans_commitments")
MEMORY_CLASSES = CORE_MEMORY_CLASSES + DYNAMIC_MEMORY_CLASSES


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    main_class: str
    sub_class: str
    text: str

    @classmethod
    def create(cls, main_class: str, sub_class: str, text: str) -> MemoryRecord:
        main_class = main_class.strip()
        sub_class = sub_class.strip()
        text = text.strip()
        return cls(_canonical_memory_id(main_class, sub_class, text), main_class, sub_class, text)


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    revision: int
    memories: tuple[MemoryRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryContext:
    core_items: tuple[MemoryRecord, ...] = ()
    retrieved_items: tuple[MemoryRecord, ...] = ()

    def render_block(self) -> str:
        memory = self.render()
        return f"# Memory\n\n{memory}" if memory else ""

    def render(self) -> str:
        lines: list[str] = []
        if self.core_items:
            lines.extend(["## Core Memory", *_format_memory_snippets(self.core_items)])
        if self.retrieved_items:
            if lines:
                lines.append("")
            lines.extend(["## Retrieved Memory", *_format_memory_snippets(self.retrieved_items)])
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DynamicMemoryHit:
    record: MemoryRecord
    sources: tuple[str, ...]
    rrf_score: float
    fts_rank: int | None = None
    vector_rank: int | None = None
    vector_similarity: float | None = None


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    database_committed: bool
    view_exported: bool
    revision: int | None = None
    error: str | None = None
    retryable: bool = True


class MemoryRevisionConflictError(RuntimeError):
    """Raised when a snapshot was based on an obsolete database revision."""


def _format_memory_snippets(items: Iterable[MemoryRecord]) -> list[str]:
    return [f"- [{item.main_class}/{item.sub_class}] {item.text}" for item in items]


def _validate_main_class(main_class: str) -> None:
    if main_class not in MEMORY_CLASSES:
        raise ValueError(f"Unsupported main_class: {main_class}")


def validate_memory_fields(main_class: str, sub_class: str, text: str) -> None:
    """Require fields that round-trip through the single-line Markdown format."""
    _validate_main_class(main_class)
    for name, value in (("sub_class", sub_class), ("memory text", text)):
        if not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        if value != value.strip() or len(value.splitlines()) != 1:
            raise ValueError(f"{name} must be a single line without surrounding whitespace")
    if ": " in sub_class:
        raise ValueError("sub_class must not contain the Markdown delimiter ': '")


def _canonical_memory_id(main_class: str, sub_class: str, text: str) -> str:
    validate_memory_fields(main_class, sub_class, text)
    safe_slot = sub_class.strip().lower()
    safe_slot = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in safe_slot)
    digest = sha1(text.strip().encode("utf-8")).hexdigest()[:12]
    return f"{main_class}:{safe_slot}:{digest}"


def _coerce_record(value: MemoryRecord | dict[str, Any]) -> MemoryRecord:
    if isinstance(value, MemoryRecord):
        record = value
    else:
        main_class = str(value.get("main_class") or "").strip()
        sub_class = str(value.get("sub_class") or "").strip()
        text = str(value.get("text") or "").strip()
        record = MemoryRecord(
            memory_id=str(
                value.get("memory_id") or _canonical_memory_id(main_class, sub_class, text)
            ),
            main_class=main_class,
            sub_class=sub_class,
            text=text,
        )
    validate_memory_fields(record.main_class, record.sub_class, record.text)
    return record


def _as_mapping(value: MemoryRecord | dict[str, Any]) -> dict[str, str]:
    record = _coerce_record(value)
    return {
        "memory_id": record.memory_id,
        "main_class": record.main_class,
        "sub_class": record.sub_class,
        "text": record.text,
    }


def render_history_markdown(events: list[dict[str, Any]]) -> str:
    ordered = sorted(
        events, key=lambda event: (str(event.get("ts") or ""), str(event.get("event_id") or ""))
    )
    entries = [str(event.get("history_text") or "").strip() for event in ordered]
    entries = [entry for entry in entries if entry]
    return "\n\n".join(entries) + ("\n\n" if entries else "")


class MemoryDatabase:
    """SQLite store. Initialization is explicit; all query methods are read-only."""

    def __init__(
        self,
        workspace: Path,
        *,
        storage_dir: Path | None = None,
        view_dir: Path | None = None,
        vec_backend: str = "sqlite-vec",
        vec_loader: Callable[[sqlite3.Connection], None] | None = None,
    ) -> None:
        if vec_backend not in {"sqlite-vec", "array"}:
            raise ValueError(f"Unsupported vector backend: {vec_backend}")
        self.memory_dir = ensure_dir(workspace / "memory")
        self.storage_dir = ensure_dir(storage_dir or self.memory_dir)
        self.view_dir = ensure_dir(view_dir or self.memory_dir)
        self.db_path = self.storage_dir / "memory.db"
        self.memory_file = self.view_dir / "MEMORY.md"
        self.history_file = self.view_dir / "HISTORY.md"
        self.vec_backend = vec_backend
        self._vec_loader = vec_loader
        self._vector_dim: int | None = None

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if self.vec_backend == "sqlite-vec":
            if self._vec_loader is not None:
                self._vec_loader(conn)
            else:
                load_sqlite_vec(conn, feature="Semantic memory")
        return conn

    def initialize(
        self,
        vector_dim: int | None = None,
        *,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Create storage once without rebuilding derived indexes on later calls."""
        with self.connect() as conn:
            conn.execute(
                "create table if not exists metadata (key text primary key, value text not null)"
            )
            version_row = conn.execute(
                "select value from metadata where key='schema_version'"
            ).fetchone()
            if version_row is not None and int(version_row["value"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Memory database schema_version={version_row['value']} is incompatible with "
                    f"schema_version={SCHEMA_VERSION}. Use scripts/upgrade_memory_knowledge.py "
                    "as described in spec/MEMORY_KNOWLEDGE_UPGRADE.md, or follow "
                    "docs/DASHBOARD_RUNTIME.md to explicitly reset disposable memory data."
                )
            conn.executescript("""
                create table if not exists raw_events (
                    event_id text primary key, ts text not null, session_key text not null,
                    history_text text not null, plain_text text not null default '',
                    main_class text, sub_class text, candidate_type text,
                    source_start_idx integer, source_end_idx integer
                );
                create table if not exists canonical_memories (
                    memory_id text primary key, main_class text not null,
                    sub_class text not null default '', text text not null
                );
                create virtual table if not exists canonical_fts using fts5(
                    memory_id UNINDEXED, tokens
                );
                create table if not exists memory_vector_map (
                    rowid integer primary key autoincrement,
                    memory_id text not null unique references canonical_memories(memory_id) on delete cascade
                );
                create table if not exists memory_sync_state (
                    singleton integer primary key check(singleton = 1),
                    published_revision integer,
                    published_text text,
                    pending_revision integer,
                    pending_expected_text text,
                    pending_target_text text
                );
            """)
            self._validate_runtime_schema(conn)
            stored = {
                key: (
                    str(row["value"])
                    if (
                        row := conn.execute(
                            "select value from metadata where key=?", (key,)
                        ).fetchone()
                    )
                    is not None
                    else None
                )
                for key in ("vector_dim", "embedding_provider", "embedding_model")
            }
            vector_dim = vector_dim or (int(stored["vector_dim"]) if stored["vector_dim"] else 1024)
            embedding_provider = embedding_provider or stored["embedding_provider"] or "dashscope"
            embedding_model = embedding_model or stored["embedding_model"] or "text-embedding-v4"
            expected_metadata = {
                "vector_dim": str(vector_dim),
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "fts_tokenizer_version": FTS_TOKENIZER_VERSION,
                "vec_backend": self.vec_backend,
            }
            for key, expected_value in expected_metadata.items():
                row = conn.execute("select value from metadata where key=?", (key,)).fetchone()
                if row is not None and str(row["value"]) != expected_value:
                    raise RuntimeError(
                        f"Memory index metadata mismatch for {key}: stored {row['value']!r}, configured {expected_value!r}. Rebuild the memory indexes."
                    )
            if self.vec_backend == "sqlite-vec":
                conn.execute(
                    f"create virtual table if not exists memory_vec using vec0(embedding float[{vector_dim}])"
                )
            else:
                conn.execute(
                    "create table if not exists memory_vec(rowid integer primary key, embedding_json text not null)"
                )
            conn.execute(
                "insert into metadata(key, value) values('schema_version', ?) on conflict(key) do update set value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
            conn.execute("insert or ignore into metadata(key, value) values('revision', '0')")
            for key, value in expected_metadata.items():
                conn.execute(
                    "insert or ignore into metadata(key, value) values(?, ?)", (key, value)
                )
            conn.execute("insert or ignore into memory_sync_state(singleton) values(1)")
            self._vector_dim = vector_dim

    @staticmethod
    def _validate_runtime_schema(conn: sqlite3.Connection) -> None:
        expected = {
            "raw_events": {
                "event_id",
                "ts",
                "session_key",
                "history_text",
                "plain_text",
                "main_class",
                "sub_class",
                "candidate_type",
                "source_start_idx",
                "source_end_idx",
            },
            "canonical_memories": {"memory_id", "main_class", "sub_class", "text"},
            "memory_vector_map": {"rowid", "memory_id"},
            "memory_sync_state": {
                "singleton",
                "published_revision",
                "published_text",
                "pending_revision",
                "pending_expected_text",
                "pending_target_text",
            },
        }
        for table, columns in expected.items():
            actual = {str(row["name"]) for row in conn.execute(f"pragma table_info({table})")}
            if actual and actual != columns:
                raise RuntimeError(
                    "Existing memory.db uses an unsupported memory schema. Back it up and run the planned data conversion before starting nanobot."
                )

    def get_schema_version(self) -> int:
        with self.connect() as conn:
            row = conn.execute("select value from metadata where key = 'schema_version'").fetchone()
        return int(row["value"]) if row else 0

    def read_snapshot(self) -> MemorySnapshot:
        with self.connect() as conn:
            revision_row = conn.execute(
                "select value from metadata where key = 'revision'"
            ).fetchone()
            rows = conn.execute(
                "select memory_id, main_class, sub_class, text from canonical_memories order by main_class, sub_class, memory_id"
            ).fetchall()
        return MemorySnapshot(
            int(revision_row["value"]) if revision_row else 0,
            tuple(MemoryRecord(**dict(row)) for row in rows),
        )

    def list_raw_events(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "select event_id, ts, session_key, history_text, plain_text, main_class, sub_class, candidate_type, source_start_idx, source_end_idx from raw_events order by ts, event_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_canonical_memories(self) -> list[dict[str, Any]]:
        return [_as_mapping(item) for item in self.read_snapshot().memories]

    def read_publish_state(self) -> dict[str, int | str | None]:
        with self.connect() as conn:
            row = conn.execute(
                "select published_revision, published_text, pending_revision, "
                "pending_expected_text, pending_target_text "
                "from memory_sync_state where singleton=1"
            ).fetchone()
        if row is None:
            raise RuntimeError("Memory sync state is not initialized")
        return dict(row)

    def read_core_memories(self) -> tuple[MemoryRecord, ...]:
        placeholders = ",".join("?" for _ in CORE_MEMORY_CLASSES)
        with self.connect() as conn:
            rows = conn.execute(
                f"select memory_id, main_class, sub_class, text from canonical_memories where main_class in ({placeholders}) order by main_class, sub_class, memory_id",
                CORE_MEMORY_CLASSES,
            ).fetchall()
        class_order = {main_class: index for index, main_class in enumerate(CORE_MEMORY_CLASSES)}
        records = [MemoryRecord(**dict(row)) for row in rows]
        return tuple(
            sorted(
                records,
                key=lambda item: (class_order[item.main_class], item.sub_class, item.memory_id),
            )
        )

    def search_dynamic_fts(self, query: str, limit: int) -> list[dict[str, Any]]:
        fts_query = build_fts_query(query.strip())
        if not fts_query or limit <= 0:
            return []
        placeholders = ",".join("?" for _ in DYNAMIC_MEMORY_CLASSES)
        with self.connect() as conn:
            rows = conn.execute(
                f"select c.memory_id, c.main_class, c.sub_class, c.text, bm25(canonical_fts) as rank_score from canonical_fts f join canonical_memories c on c.memory_id=f.memory_id where canonical_fts match ? and c.main_class in ({placeholders}) order by bm25(canonical_fts), c.memory_id limit ?",
                (fts_query, *DYNAMIC_MEMORY_CLASSES, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_dynamic_vector(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        if self._vector_dim is None or len(query_vector) != self._vector_dim:
            raise RuntimeError(
                f"Memory query vector dimension mismatch: expected {self._vector_dim}, got {len(query_vector)}"
            )
        if self.vec_backend == "sqlite-vec":
            with self.connect() as conn:
                rows = conn.execute(
                    """with hits as (select rowid, distance from memory_vec where embedding match ? and k = ?)
                    select c.memory_id, c.main_class, c.sub_class, c.text, hits.distance as rank_score,
                           1.0 - (hits.distance * hits.distance / 2.0) as vector_similarity
                    from hits join memory_vector_map m on m.rowid=hits.rowid
                    join canonical_memories c on c.memory_id=m.memory_id
                    order by hits.distance, c.memory_id limit ?""",
                    (vector_json(query_vector), limit, limit),
                ).fetchall()
            return [dict(row) for row in rows]
        with self.connect() as conn:
            rows = conn.execute(
                "select c.memory_id, c.main_class, c.sub_class, c.text, v.embedding_json "
                "from memory_vec v join memory_vector_map m on m.rowid=v.rowid "
                "join canonical_memories c on c.memory_id=m.memory_id"
            ).fetchall()
        hits = []
        for row in rows:
            similarity = cosine_similarity(
                query_vector, [float(v) for v in json.loads(row["embedding_json"])]
            )
            item = dict(row)
            item.pop("embedding_json")
            item.update(rank_score=1.0 - similarity, vector_similarity=similarity)
            hits.append(item)
        hits.sort(key=lambda item: (item["rank_score"], item["memory_id"]))
        return hits[:limit]

    @staticmethod
    def fuse_dynamic(
        result_sets: dict[str, list[dict[str, Any]]], limit: int
    ) -> list[dict[str, Any]]:
        return reciprocal_rank_fuse(result_sets, key_field="memory_id", limit=limit)

    def query_dynamic_memories(self, query: str, limit: int = 5) -> tuple[MemoryRecord, ...]:
        return tuple(
            MemoryRecord(item["memory_id"], item["main_class"], item["sub_class"], item["text"])
            for item in self.search_dynamic_fts(query, limit)
        )

    def query_canonical_memories(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return [_as_mapping(item) for item in self.query_dynamic_memories(query, limit)]

    def rebuild_indexes(
        self,
        *,
        dynamic_embeddings: dict[str, list[float]],
        vector_dim: int,
        embedding_provider: str,
        embedding_model: str,
    ) -> None:
        """Rebuild only derived FTS/vector state from canonical memories."""
        snapshot = self.read_snapshot()
        dynamic = tuple(
            item for item in snapshot.memories if item.main_class in DYNAMIC_MEMORY_CLASSES
        )
        if set(dynamic_embeddings) != {item.memory_id for item in dynamic}:
            raise RuntimeError("Rebuild requires one embedding for every dynamic memory")
        if any(len(vector) != vector_dim for vector in dynamic_embeddings.values()):
            raise RuntimeError("Rebuild embedding dimension does not match vector_dim")
        with self.connect() as conn:
            conn.execute("begin immediate")
            conn.execute("delete from canonical_fts")
            conn.execute("drop table if exists memory_vec")
            conn.execute("delete from memory_vector_map")
            if self.vec_backend == "sqlite-vec":
                conn.execute(
                    f"create virtual table memory_vec using vec0(embedding float[{vector_dim}])"
                )
            else:
                conn.execute(
                    "create table memory_vec(rowid integer primary key, embedding_json text not null)"
                )
            for record in dynamic:
                conn.execute(
                    "insert into canonical_fts(memory_id, tokens) values (?, ?)",
                    (record.memory_id, fts_index_text(f"{record.sub_class} {record.text}")),
                )
                conn.execute(
                    "insert into memory_vector_map(memory_id) values (?)", (record.memory_id,)
                )
                rowid = int(
                    conn.execute(
                        "select rowid from memory_vector_map where memory_id=?",
                        (record.memory_id,),
                    ).fetchone()["rowid"]
                )
                vector = vector_json(dynamic_embeddings[record.memory_id])
                column = "embedding" if self.vec_backend == "sqlite-vec" else "embedding_json"
                conn.execute(
                    f"insert into memory_vec(rowid, {column}) values (?, ?)", (rowid, vector)
                )
            metadata = {
                "vector_dim": str(vector_dim),
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "fts_tokenizer_version": FTS_TOKENIZER_VERSION,
                "vec_backend": self.vec_backend,
            }
            for key, value in metadata.items():
                conn.execute(
                    "insert into metadata(key, value) values(?, ?) "
                    "on conflict(key) do update set value=excluded.value",
                    (key, value),
                )
        self._vector_dim = vector_dim

    def commit_snapshot(
        self,
        snapshot: MemorySnapshot,
        *,
        expected_revision: int,
        event_id: str,
        ts: str,
        session_key: str,
        history_text: str,
        plain_text: str = "",
        candidate_type: str = "snapshot",
        publish_expected_text: str | None = None,
        publish_target_text: str | None = None,
        stage_publish: bool = False,
        dynamic_embeddings: dict[str, list[float]] | None = None,
    ) -> int:
        """Atomically persist the event, full snapshot, FTS rows, and next revision."""
        if snapshot.revision != expected_revision:
            raise MemoryRevisionConflictError(
                f"Snapshot revision {snapshot.revision} does not match expected revision {expected_revision}"
            )
        records = tuple(_coerce_record(item) for item in snapshot.memories)
        if len({item.memory_id for item in records}) != len(records):
            raise sqlite3.IntegrityError("duplicate memory_id in snapshot")
        with self.connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute("select value from metadata where key='revision'").fetchone()
            current_revision = int(row["value"]) if row else 0
            if current_revision != expected_revision:
                raise MemoryRevisionConflictError(
                    f"Memory revision changed: expected {expected_revision}, found {current_revision}"
                )
            conn.execute(
                "insert into raw_events(event_id, ts, session_key, history_text, plain_text, candidate_type) values (?, ?, ?, ?, ?, ?)",
                (event_id, ts, session_key, history_text, plain_text, candidate_type),
            )
            existing_rows = conn.execute(
                "select memory_id, main_class, sub_class, text from canonical_memories"
            ).fetchall()
            existing = {str(row["memory_id"]): tuple(row) for row in existing_rows}
            target = {record.memory_id: record for record in records}
            removed_ids = set(existing) - set(target)
            changed = [
                record
                for record in records
                if existing.get(record.memory_id)
                != (record.memory_id, record.main_class, record.sub_class, record.text)
            ]
            dynamic_changed = [
                record for record in changed if record.main_class in DYNAMIC_MEMORY_CLASSES
            ]
            embeddings = dynamic_embeddings or {}
            if {record.memory_id for record in dynamic_changed} != set(embeddings):
                raise RuntimeError(
                    "Embeddings are required for every added or changed dynamic memory"
                )
            for memory_id in removed_ids:
                mapping = conn.execute(
                    "select rowid from memory_vector_map where memory_id=?", (memory_id,)
                ).fetchone()
                if mapping is not None:
                    conn.execute("delete from memory_vec where rowid=?", (int(mapping["rowid"]),))
                conn.execute("delete from canonical_fts where memory_id=?", (memory_id,))
                conn.execute("delete from canonical_memories where memory_id=?", (memory_id,))
            for record in changed:
                values = (record.memory_id, record.main_class, record.sub_class, record.text)
                conn.execute(
                    "insert into canonical_memories(memory_id, main_class, sub_class, text) values (?, ?, ?, ?) on conflict(memory_id) do update set main_class=excluded.main_class, sub_class=excluded.sub_class, text=excluded.text",
                    values,
                )
                conn.execute("delete from canonical_fts where memory_id=?", (record.memory_id,))
                if record.main_class not in DYNAMIC_MEMORY_CLASSES:
                    mapping = conn.execute(
                        "select rowid from memory_vector_map where memory_id=?", (record.memory_id,)
                    ).fetchone()
                    if mapping is not None:
                        conn.execute(
                            "delete from memory_vec where rowid=?", (int(mapping["rowid"]),)
                        )
                        conn.execute(
                            "delete from memory_vector_map where memory_id=?", (record.memory_id,)
                        )
                    continue
                conn.execute(
                    "insert into canonical_fts(memory_id, tokens) values (?, ?)",
                    (record.memory_id, fts_index_text(f"{record.sub_class} {record.text}")),
                )
                conn.execute(
                    "insert or ignore into memory_vector_map(memory_id) values (?)",
                    (record.memory_id,),
                )
                rowid = int(
                    conn.execute(
                        "select rowid from memory_vector_map where memory_id=?", (record.memory_id,)
                    ).fetchone()["rowid"]
                )
                vector = embeddings[record.memory_id]
                if len(vector) != self._vector_dim:
                    raise RuntimeError(
                        f"Memory embedding dimension mismatch for {record.memory_id}"
                    )
                conn.execute("delete from memory_vec where rowid=?", (rowid,))
                if self.vec_backend == "sqlite-vec":
                    conn.execute(
                        "insert into memory_vec(rowid, embedding) values (?, ?)",
                        (rowid, vector_json(vector)),
                    )
                else:
                    conn.execute(
                        "insert into memory_vec(rowid, embedding_json) values (?, ?)",
                        (rowid, vector_json(vector)),
                    )
            new_revision = current_revision + 1
            conn.execute("update metadata set value=? where key='revision'", (str(new_revision),))
            if stage_publish:
                if publish_target_text is None:
                    raise ValueError("publish_target_text is required when stage_publish is true")
                pending = conn.execute(
                    "select pending_revision from memory_sync_state where singleton=1"
                ).fetchone()
                if pending is None or pending["pending_revision"] is not None:
                    raise RuntimeError("A memory view publication is already pending")
                conn.execute(
                    "update memory_sync_state set pending_revision=?, pending_expected_text=?, "
                    "pending_target_text=? where singleton=1",
                    (new_revision, publish_expected_text, publish_target_text),
                )
        return new_revision

    def stage_memory_publish(
        self,
        *,
        revision: int,
        expected_text: str | None,
        target_text: str,
    ) -> None:
        """Record a publication intent without changing the memory revision."""
        with self.connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute("select value from metadata where key='revision'").fetchone()
            current_revision = int(row["value"]) if row else 0
            if current_revision != revision:
                raise MemoryRevisionConflictError(
                    f"Memory revision changed: expected {revision}, found {current_revision}"
                )
            state = conn.execute(
                "select pending_revision from memory_sync_state where singleton=1"
            ).fetchone()
            if state is None or state["pending_revision"] is not None:
                raise RuntimeError("A memory view publication is already pending")
            conn.execute(
                "update memory_sync_state set pending_revision=?, pending_expected_text=?, "
                "pending_target_text=? where singleton=1",
                (revision, expected_text, target_text),
            )

    def confirm_memory_publish(self, *, revision: int, target_text: str) -> None:
        with self.connect() as conn:
            conn.execute("begin immediate")
            state = conn.execute(
                "select pending_revision, pending_target_text from memory_sync_state where singleton=1"
            ).fetchone()
            if (
                state is None
                or state["pending_revision"] != revision
                or state["pending_target_text"] != target_text
            ):
                raise MemoryRevisionConflictError(
                    "Pending memory publication changed before confirmation"
                )
            conn.execute(
                "update memory_sync_state set published_revision=?, published_text=?, "
                "pending_revision=null, pending_expected_text=null, pending_target_text=null "
                "where singleton=1",
                (revision, target_text),
            )

    def record_memory_publish_baseline(self, *, revision: int, text: str) -> None:
        """Adopt an exact existing canonical view as the initial synchronization baseline."""
        with self.connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute("select value from metadata where key='revision'").fetchone()
            current_revision = int(row["value"]) if row else 0
            state = conn.execute(
                "select published_revision, pending_revision from memory_sync_state where singleton=1"
            ).fetchone()
            if current_revision != revision:
                raise MemoryRevisionConflictError(
                    f"Memory revision changed: expected {revision}, found {current_revision}"
                )
            if (
                state is None
                or state["published_revision"] is not None
                or state["pending_revision"] is not None
            ):
                raise MemoryRevisionConflictError(
                    "Memory publication baseline is already initialized"
                )
            conn.execute(
                "update memory_sync_state set published_revision=?, published_text=? where singleton=1",
                (revision, text),
            )

    def render_history_view(self) -> str:
        return render_history_markdown(self.list_raw_events())
