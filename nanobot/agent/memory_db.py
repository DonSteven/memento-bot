"""SQLite persistence and data contracts for structured personal memory."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

from nanobot.utils.helpers import ensure_dir

SCHEMA_VERSION = 6
CORE_MEMORY_CLASSES = ("personal_profile", "preferences", "constraints")
DYNAMIC_MEMORY_CLASSES = ("projects", "daily_life", "plans_commitments")
MEMORY_CLASSES = CORE_MEMORY_CLASSES + DYNAMIC_MEMORY_CLASSES

_SECTION_ORDER = (
    ("personal_profile", "Personal Profile", "(Stable background information about the user)"),
    ("preferences", "Preferences", "(How the user prefers to communicate and collaborate)"),
    ("constraints", "Constraints", "(Rules, boundaries, and requirements that must be respected)"),
    ("projects", "Projects", "(Information about ongoing learning and work projects)"),
    ("daily_life", "Daily Life", "(Daily routines, interests, and hobbies)"),
    ("plans_commitments", "Plans and Commitments", "(Future plans, commitments, deadlines, and to-dos)"),
)
_SECTION_INDEX = {key: (title, placeholder) for key, title, placeholder in _SECTION_ORDER}
_TITLE_TO_MAIN_CLASS = {title: key for key, title, _ in _SECTION_ORDER}


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
class MemoryWriteResult:
    database_committed: bool
    view_exported: bool
    revision: int | None = None
    error: str | None = None


class MemoryRevisionConflictError(RuntimeError):
    """Raised when a snapshot was based on an obsolete database revision."""


def _format_memory_snippets(items: Iterable[MemoryRecord]) -> list[str]:
    return [f"- [{item.main_class}/{item.sub_class}] {item.text}" for item in items]


def _validate_main_class(main_class: str) -> None:
    if main_class not in _SECTION_INDEX:
        raise ValueError(f"Unsupported main_class: {main_class}")


def _validate_sub_class(sub_class: str) -> None:
    if not sub_class.strip():
        raise ValueError("sub_class must be a non-empty string")


def _canonical_memory_id(main_class: str, sub_class: str, text: str) -> str:
    _validate_main_class(main_class)
    _validate_sub_class(sub_class)
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
            memory_id=str(value.get("memory_id") or _canonical_memory_id(main_class, sub_class, text)),
            main_class=main_class,
            sub_class=sub_class,
            text=text,
        )
    _validate_main_class(record.main_class)
    _validate_sub_class(record.sub_class)
    if not record.text.strip():
        raise ValueError("memory text must be a non-empty string")
    return record


def _as_mapping(value: MemoryRecord | dict[str, Any]) -> dict[str, str]:
    record = _coerce_record(value)
    return {"memory_id": record.memory_id, "main_class": record.main_class, "sub_class": record.sub_class, "text": record.text}


def render_memory_markdown(memories: Iterable[MemoryRecord | dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, str]]] = {key: [] for key in MEMORY_CLASSES}
    for value in memories:
        if isinstance(value, MemoryRecord):
            memory = {
                "memory_id": value.memory_id,
                "main_class": value.main_class,
                "sub_class": value.sub_class.strip(),
                "text": value.text.strip(),
            }
        else:
            memory = {
                "memory_id": str(value.get("memory_id") or ""),
                "main_class": str(value.get("main_class") or ""),
                "sub_class": str(value.get("sub_class") or "").strip(),
                "text": str(value.get("text") or "").strip(),
            }
        _validate_main_class(memory["main_class"])
        if not memory["sub_class"] or not memory["text"]:
            continue
        grouped[memory["main_class"]].append(memory)
    lines = ["# Long-term Memory", "", "This file stores important information that should persist across sessions.", ""]
    for main_class, title, placeholder in _SECTION_ORDER:
        lines.extend([f"## {title}", ""])
        items = sorted(grouped[main_class], key=lambda item: (item["sub_class"], item["memory_id"], item["text"]))
        lines.extend(f"- {item['sub_class']}: {item['text']}" for item in items) if items else lines.append(placeholder)
        lines.append("")
    lines.extend(["---", "", "*This file is automatically updated by nanobot when important information should be remembered.*"])
    return "\n".join(lines)


def render_history_markdown(events: list[dict[str, Any]]) -> str:
    ordered = sorted(events, key=lambda event: (str(event.get("ts") or ""), str(event.get("event_id") or "")))
    entries = [str(event.get("history_text") or "").strip() for event in ordered]
    entries = [entry for entry in entries if entry]
    return "\n\n".join(entries) + ("\n\n" if entries else "")


def parse_memory_markdown(content: str) -> list[dict[str, Any]]:
    """Parse the canonical Markdown view. Strict validation is introduced in P2."""
    memories: list[dict[str, Any]] = []
    current_main_class: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current_main_class = _TITLE_TO_MAIN_CLASS.get(line[3:].strip())
            continue
        if current_main_class is None:
            continue
        placeholder = _SECTION_INDEX[current_main_class][1]
        if line == placeholder or line.startswith("---") or line.startswith("*This file is automatically"):
            continue
        if not line.startswith("- ") or ": " not in line[2:]:
            continue
        sub_class, text = (part.strip() for part in line[2:].split(": ", 1))
        if not sub_class or not text:
            continue
        memories.append({"memory_id": _canonical_memory_id(current_main_class, sub_class, text), "main_class": current_main_class, "sub_class": sub_class, "text": text})
    return memories


def _build_fts_query(query: str) -> str:
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
            normalized = f"{token}*"
            if len(token) >= 3 and normalized not in seen:
                seen.add(normalized)
                terms.append(normalized)
    return " OR ".join(terms)


class MemoryDatabase:
    """SQLite store. Initialization is explicit; all query methods are read-only."""

    def __init__(self, workspace: Path, *, storage_dir: Path | None = None, view_dir: Path | None = None) -> None:
        self.memory_dir = ensure_dir(workspace / "memory")
        self.storage_dir = ensure_dir(storage_dir or self.memory_dir)
        self.view_dir = ensure_dir(view_dir or self.memory_dir)
        self.db_path = self.storage_dir / "memory.db"
        self.memory_file = self.view_dir / "MEMORY.md"
        self.history_file = self.view_dir / "HISTORY.md"

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        """Create storage once without rebuilding derived indexes on later calls."""
        with self.connect() as conn:
            conn.executescript("""
                create table if not exists metadata (key text primary key, value text not null);
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
                    memory_id UNINDEXED, main_class UNINDEXED, sub_class, text
                );
            """)
            self._validate_runtime_schema(conn)
            conn.execute("insert into metadata(key, value) values('schema_version', ?) on conflict(key) do update set value=excluded.value", (str(SCHEMA_VERSION),))
            conn.execute("insert or ignore into metadata(key, value) values('revision', '0')")

    @staticmethod
    def _validate_runtime_schema(conn: sqlite3.Connection) -> None:
        expected = {
            "raw_events": {"event_id", "ts", "session_key", "history_text", "plain_text", "main_class", "sub_class", "candidate_type", "source_start_idx", "source_end_idx"},
            "canonical_memories": {"memory_id", "main_class", "sub_class", "text"},
        }
        for table, columns in expected.items():
            actual = {str(row["name"]) for row in conn.execute(f"pragma table_info({table})")}
            if actual and actual != columns:
                raise RuntimeError("Existing memory.db uses an unsupported memory schema. Back it up and run the planned data conversion before starting nanobot.")

    def get_schema_version(self) -> int:
        with self.connect() as conn:
            row = conn.execute("select value from metadata where key = 'schema_version'").fetchone()
        return int(row["value"]) if row else 0

    def read_snapshot(self) -> MemorySnapshot:
        with self.connect() as conn:
            revision_row = conn.execute("select value from metadata where key = 'revision'").fetchone()
            rows = conn.execute("select memory_id, main_class, sub_class, text from canonical_memories order by main_class, sub_class, memory_id").fetchall()
        return MemorySnapshot(int(revision_row["value"]) if revision_row else 0, tuple(MemoryRecord(**dict(row)) for row in rows))

    def list_raw_events(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select event_id, ts, session_key, history_text, plain_text, main_class, sub_class, candidate_type, source_start_idx, source_end_idx from raw_events order by ts, event_id").fetchall()
        return [dict(row) for row in rows]

    def list_canonical_memories(self) -> list[dict[str, Any]]:
        return [_as_mapping(item) for item in self.read_snapshot().memories]

    def read_core_memories(self) -> tuple[MemoryRecord, ...]:
        placeholders = ",".join("?" for _ in CORE_MEMORY_CLASSES)
        with self.connect() as conn:
            rows = conn.execute(f"select memory_id, main_class, sub_class, text from canonical_memories where main_class in ({placeholders}) order by main_class, sub_class, memory_id", CORE_MEMORY_CLASSES).fetchall()
        class_order = {main_class: index for index, main_class in enumerate(CORE_MEMORY_CLASSES)}
        records = [MemoryRecord(**dict(row)) for row in rows]
        return tuple(sorted(records, key=lambda item: (class_order[item.main_class], item.sub_class, item.memory_id)))

    def query_dynamic_memories(self, query: str, limit: int = 5) -> tuple[MemoryRecord, ...]:
        fts_query = _build_fts_query(query.strip())
        if not fts_query or limit <= 0:
            return ()
        placeholders = ",".join("?" for _ in DYNAMIC_MEMORY_CLASSES)
        with self.connect() as conn:
            rows = conn.execute(
                f"select c.memory_id, c.main_class, c.sub_class, c.text from canonical_fts f join canonical_memories c on c.memory_id=f.memory_id where canonical_fts match ? and c.main_class in ({placeholders}) order by bm25(canonical_fts), c.main_class, c.sub_class, c.memory_id limit ?",
                (fts_query, *DYNAMIC_MEMORY_CLASSES, limit),
            ).fetchall()
        return tuple(MemoryRecord(**dict(row)) for row in rows)

    def query_canonical_memories(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return [_as_mapping(item) for item in self.query_dynamic_memories(query, limit)]

    def commit_snapshot(self, snapshot: MemorySnapshot, *, expected_revision: int, event_id: str, ts: str, session_key: str, history_text: str, plain_text: str = "", candidate_type: str = "snapshot") -> int:
        """Atomically persist the event, full snapshot, FTS rows, and next revision."""
        if snapshot.revision != expected_revision:
            raise MemoryRevisionConflictError(
                f"Snapshot revision {snapshot.revision} does not match expected revision {expected_revision}"
            )
        records = tuple(_coerce_record(item) for item in snapshot.memories)
        with self.connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute("select value from metadata where key='revision'").fetchone()
            current_revision = int(row["value"]) if row else 0
            if current_revision != expected_revision:
                raise MemoryRevisionConflictError(f"Memory revision changed: expected {expected_revision}, found {current_revision}")
            conn.execute("insert into raw_events(event_id, ts, session_key, history_text, plain_text, candidate_type) values (?, ?, ?, ?, ?, ?)", (event_id, ts, session_key, history_text, plain_text, candidate_type))
            conn.execute("delete from canonical_memories")
            conn.execute("delete from canonical_fts")
            for record in records:
                values = (record.memory_id, record.main_class, record.sub_class, record.text)
                conn.execute("insert into canonical_memories(memory_id, main_class, sub_class, text) values (?, ?, ?, ?)", values)
                conn.execute("insert into canonical_fts(memory_id, main_class, sub_class, text) values (?, ?, ?, ?)", values)
            new_revision = current_revision + 1
            conn.execute("update metadata set value=? where key='revision'", (str(new_revision),))
        return new_revision

    def render_memory_view(self) -> str:
        return render_memory_markdown(self.read_snapshot().memories)

    def render_history_view(self) -> str:
        return render_history_markdown(self.list_raw_events())

    def write_views(self) -> None:
        self.memory_file.write_text(self.render_memory_view(), encoding="utf-8")
        self.history_file.write_text(self.render_history_view(), encoding="utf-8")
