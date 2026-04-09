"""SQLite storage base for the next-generation memory system."""

from __future__ import annotations

import re
import sqlite3
from hashlib import sha1
from pathlib import Path
from typing import Any

from nanobot.utils.helpers import ensure_dir


SCHEMA_VERSION = 5

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


def _validate_main_class(main_class: str) -> None:
    if main_class not in _SECTION_INDEX:
        raise ValueError(f"Unsupported main_class: {main_class}")


def _validate_sub_class(sub_class: str) -> None:
    if not sub_class.strip():
        raise ValueError("sub_class must be a non-empty string")


def render_memory_markdown(memories: list[dict[str, Any]]) -> str:
    """Render canonical memories into the legacy MEMORY.md view.
        把“结构化的长期记忆列表”渲染成一个给人看的 MEMORY.md 文本"""
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key, _, _ in _SECTION_ORDER}
    for memory in memories:
        main_class = str(memory["main_class"])
        _validate_main_class(main_class)
        grouped[main_class].append(memory)

    lines = [
        "# Long-term Memory",
        "",
        "This file stores important information that should persist across sessions.",
        "",
    ]

    for main_class, title, placeholder in _SECTION_ORDER:
        lines.append(f"## {title}")
        lines.append("")

        section_items = sorted(
            grouped[main_class],
            key=lambda item: (
                str(item.get("sub_class") or ""),
                str(item.get("memory_id") or ""),
                str(item.get("text") or ""),
            ),
        )
        rendered_items: list[str] = []
        for item in section_items:
            sub_class = str(item.get("sub_class") or "").strip()
            text = str(item.get("text") or "").strip()
            if not sub_class or not text:
                continue
            rendered_items.append(f"- {sub_class}: {text}")

        if not rendered_items:
            lines.append(placeholder)
            lines.append("")
            continue

        for rendered_item in rendered_items:
            lines.append(rendered_item)
        lines.append("")

    lines.extend([
        "---",
        "",
        "*This file is automatically updated by nanobot when important information should be remembered.*",
    ])
    return "\n".join(lines)


def render_history_markdown(events: list[dict[str, Any]]) -> str:
    """Render raw events into the legacy HISTORY.md view."""
    ordered = sorted(
        events,
        key=lambda event: (
            str(event.get("ts") or ""),
            str(event.get("event_id") or ""),
        ),
    )
    entries = [str(event.get("history_text") or "").strip() for event in ordered]
    entries = [entry for entry in entries if entry]
    if not entries:
        return ""
    return "\n\n".join(entries) + "\n\n"


def _canonical_memory_id(main_class: str, sub_class: str, text: str) -> str:
    _validate_sub_class(sub_class)
    safe_slot = sub_class.strip().lower()
    safe_slot = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in safe_slot)
    digest = sha1(text.strip().encode("utf-8")).hexdigest()[:12]
    return f"{main_class}:{safe_slot}:{digest}"


def _build_fts_query(query: str) -> str:
    """Build a tolerant FTS query from user text."""
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


def parse_memory_markdown(content: str) -> list[dict[str, Any]]:
    """Parse a legacy MEMORY.md view into canonical memory items."""
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
        if not line.startswith("- "):
            continue

        body = line[2:].strip()
        if not body:
            continue
        if ": " not in body:
            continue
        maybe_sub_class, maybe_text = body.split(": ", 1)
        sub_class = maybe_sub_class.strip()
        text = maybe_text.strip()
        if not sub_class or not text:
            continue
        memory_id = _canonical_memory_id(current_main_class, sub_class, text)
        memories.append({
            "memory_id": memory_id,
            "main_class": current_main_class,
            "sub_class": sub_class,
            "text": text,
        })
    return memories


class MemoryDatabase:
    """Minimal SQLite storage base for raw events and canonical memories."""

    def __init__(
        self,
        workspace: Path,
        *,
        storage_dir: Path | None = None,
        view_dir: Path | None = None,
    ):
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
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists metadata (
                    key text primary key,
                    value text not null
                );

                create table if not exists raw_events (
                    event_id text primary key,
                    ts text not null,
                    session_key text not null,
                    history_text text not null,
                    plain_text text not null default '',
                    main_class text,
                    sub_class text,
                    candidate_type text,
                    source_start_idx integer,
                    source_end_idx integer
                );

                create table if not exists canonical_memories (
                    memory_id text primary key,
                    main_class text not null,
                    sub_class text not null default '',
                    text text not null
                );

                drop table if exists memory_evidence;

                create virtual table if not exists canonical_fts using fts5(
                    memory_id UNINDEXED,
                    main_class UNINDEXED,
                    sub_class,
                    text
                );
                """
            )
            self._validate_runtime_schema(conn)
            conn.execute(
                """
                insert into metadata(key, value) values('schema_version', ?)
                on conflict(key) do update set value=excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            self._rebuild_fts(conn)

    @staticmethod
    def _validate_runtime_schema(conn: sqlite3.Connection) -> None:
        required_columns = {
            "raw_events": {
                "event_id", "ts", "session_key", "history_text", "plain_text",
                "main_class", "sub_class", "candidate_type",
                "source_start_idx", "source_end_idx",
            },
            "canonical_memories": {"memory_id", "main_class", "sub_class", "text"},
        }
        for table_name, expected in required_columns.items():
            columns = {
                str(row["name"])
                for row in conn.execute(f"pragma table_info({table_name})").fetchall()
            }
            if not columns:
                continue
            if "confidence" in columns or columns != expected:
                raise RuntimeError(
                    "Existing memory.db uses an unsupported memory schema. "
                    "Delete memory/memory.db manually and rerun nanobot."
                )

    @staticmethod
    def _rebuild_fts(conn: sqlite3.Connection) -> None:
        conn.execute("delete from canonical_fts")
        conn.execute(
            """
            insert into canonical_fts(memory_id, main_class, sub_class, text)
            select memory_id, main_class, sub_class, text
            from canonical_memories
            """
        )

    def get_schema_version(self) -> int:
        self.initialize()
        with self.connect() as conn:
            row = conn.execute(
                "select value from metadata where key = 'schema_version'"
            ).fetchone()
        return int(row["value"]) if row else 0

    def insert_raw_event(
        self,
        *,
        event_id: str,
        ts: str,
        session_key: str,
        history_text: str,
        plain_text: str = "",
        main_class: str | None = None,
        sub_class: str | None = None,
        candidate_type: str | None = None,
        source_start_idx: int | None = None,
        source_end_idx: int | None = None,
    ) -> None:
        self.initialize()
        if main_class is not None:
            _validate_main_class(main_class)
        with self.connect() as conn:
            self._insert_raw_event_row(
                conn,
                event_id=event_id,
                ts=ts,
                session_key=session_key,
                history_text=history_text,
                plain_text=plain_text,
                main_class=main_class,
                sub_class=sub_class,
                candidate_type=candidate_type,
                source_start_idx=source_start_idx,
                source_end_idx=source_end_idx,
            )

    def upsert_canonical_memory(
        self,
        *,
        memory_id: str,
        main_class: str,
        text: str,
        sub_class: str,
    ) -> None:
        self.initialize()
        _validate_main_class(main_class)
        _validate_sub_class(sub_class)
        with self.connect() as conn:
            conn.execute(
                """
                insert into canonical_memories(
                    memory_id, main_class, sub_class, text
                ) values (?, ?, ?, ?)
                on conflict(memory_id) do update set
                    main_class=excluded.main_class,
                    sub_class=excluded.sub_class,
                    text=excluded.text
                """,
                (
                    memory_id,
                    main_class,
                    sub_class,
                    text,
                ),
            )
            self._rebuild_fts(conn)

    def list_raw_events(self) -> list[dict[str, Any]]:
        self.initialize()
        with self.connect() as conn:
            rows = conn.execute(
                """
                select event_id, ts, session_key, history_text, plain_text,
                       main_class, sub_class, candidate_type,
                       source_start_idx, source_end_idx
                from raw_events
                order by ts asc, event_id asc
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_canonical_memories(self) -> list[dict[str, Any]]:
        self.initialize()
        with self.connect() as conn:
            rows = conn.execute(
                """
                select memory_id, main_class, sub_class, text
                from canonical_memories
                order by main_class asc, sub_class asc, memory_id asc
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def render_memory_view(self) -> str:
        return render_memory_markdown(self.list_canonical_memories())

    def render_history_view(self) -> str:
        return render_history_markdown(self.list_raw_events())

    def write_views(self) -> None:
        self.initialize()
        self.memory_file.write_text(self.render_memory_view(), encoding="utf-8")
        self.history_file.write_text(self.render_history_view(), encoding="utf-8")

    def replace_canonical_snapshot(
        self,
        memories: list[dict[str, Any]],
    ) -> None:
        """Replace the canonical snapshot with a new set of canonical memories."""
        self.initialize()
        with self.connect() as conn:
            self._replace_canonical_snapshot_rows(conn, memories)

    def write_structured_snapshot(
        self,
        *,
        event_id: str,
        ts: str,
        session_key: str,
        history_text: str,
        plain_text: str = "",
        candidate_type: str | None = None,
        memories: list[dict[str, Any]],
    ) -> None:
        """Write one raw event plus the full canonical snapshot before re-rendering views."""
        self.initialize()
        with self.connect() as conn:
            self._insert_raw_event_row(
                conn,
                event_id=event_id,
                ts=ts,
                session_key=session_key,
                history_text=history_text,
                plain_text=plain_text,
                candidate_type=candidate_type,
            )
            self._replace_canonical_snapshot_rows(conn, memories)
        self.write_views()

    @staticmethod
    def _insert_raw_event_row(
        conn: sqlite3.Connection,
        *,
        event_id: str,
        ts: str,
        session_key: str,
        history_text: str,
        plain_text: str = "",
        main_class: str | None = None,
        sub_class: str | None = None,
        candidate_type: str | None = None,
        source_start_idx: int | None = None,
        source_end_idx: int | None = None,
    ) -> None:
        if main_class is not None:
            _validate_main_class(main_class)
        conn.execute(
            """
            insert into raw_events(
                event_id, ts, session_key, history_text, plain_text,
                main_class, sub_class, candidate_type,
                source_start_idx, source_end_idx
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                ts,
                session_key,
                history_text,
                plain_text,
                main_class,
                sub_class,
                candidate_type,
                source_start_idx,
                source_end_idx,
            ),
        )

    @staticmethod
    def _replace_canonical_snapshot_rows(
        conn: sqlite3.Connection,
        memories: list[dict[str, Any]],
    ) -> None:
        conn.execute("delete from canonical_memories")
        for memory in memories:
            main_class = str(memory["main_class"])
            _validate_main_class(main_class)
            sub_class = str(memory.get("sub_class") or "").strip()
            _validate_sub_class(sub_class)
            text = str(memory.get("text") or "")
            conn.execute(
                """
                insert into canonical_memories(
                    memory_id, main_class, sub_class, text
                ) values (?, ?, ?, ?)
                """,
                (
                    str(memory.get("memory_id") or _canonical_memory_id(main_class, sub_class, text)),
                    main_class,
                    sub_class,
                    text,
                ),
            )
        MemoryDatabase._rebuild_fts(conn)

    def query_canonical_memories(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve canonical memories with SQLite FTS5."""
        self.initialize()
        query = query.strip()
        if not query:
            return []
        fts_query = _build_fts_query(query)
        if not fts_query:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                select c.memory_id, c.main_class, c.sub_class, c.text
                from canonical_fts f
                join canonical_memories c on c.memory_id = f.memory_id
                where canonical_fts match ?
                order by bm25(canonical_fts), c.main_class asc, c.sub_class asc, c.memory_id asc
                limit ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [dict(row) for row in rows]
