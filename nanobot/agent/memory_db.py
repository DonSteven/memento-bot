"""SQLite storage base for the next-generation memory system."""

from __future__ import annotations

import json
import re
import sqlite3
from hashlib import sha1
from pathlib import Path
from typing import Any

from nanobot.utils.helpers import ensure_dir


SCHEMA_VERSION = 1

_SECTION_ORDER = (
    ("user_info", "User Information", "(Important facts about the user)"),
    ("preferences", "Preferences", "(User preferences learned over time)"),
    ("project_context", "Project Context", "(Information about ongoing projects)"),
    ("important_notes", "Important Notes", "(Things to remember)"),
)
_SECTION_INDEX = {key: (title, placeholder) for key, title, placeholder in _SECTION_ORDER}
_TITLE_TO_MAIN_CLASS = {title: key for key, title, _ in _SECTION_ORDER}


def _validate_main_class(main_class: str) -> None:
    if main_class not in _SECTION_INDEX:
        raise ValueError(f"Unsupported main_class: {main_class}")


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
            if not text:
                continue
            if sub_class and sub_class != "general":
                rendered_items.append(f"- {sub_class}: {text}")
            else:
                rendered_items.append(f"- {text}")

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
    return "\n\n".join(entries)


def _canonical_memory_id(main_class: str, sub_class: str, text: str) -> str:
    safe_slot = (sub_class or "general").strip().lower() or "general"
    safe_slot = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in safe_slot)
    digest = sha1(text.strip().encode("utf-8")).hexdigest()[:12]
    return f"{main_class}:{safe_slot}:{digest}"




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
        sub_class = ""
        text = body
        if ": " in body:
            maybe_sub_class, maybe_text = body.split(": ", 1)
            if maybe_text.strip():
                sub_class = maybe_sub_class.strip()
                text = maybe_text.strip()
        memory_id = _canonical_memory_id(current_main_class, sub_class, text)
        memories.append({
            "memory_id": memory_id,
            "main_class": current_main_class,
            "sub_class": sub_class,
            "text": text,
        })
    return memories


class MemoryDatabase:
    """Minimal SQLite storage base for raw events, canonical memories, and evidence links."""

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
                    extracted_json text,
                    main_class text,
                    sub_class text,
                    candidate_type text,
                    confidence real not null default 0.0,
                    source_start_idx integer,
                    source_end_idx integer
                );

                create table if not exists canonical_memories (
                    memory_id text primary key,
                    main_class text not null,
                    sub_class text not null default '',
                    text text not null,
                    confidence real not null default 0.0,
                    priority real not null default 0.0,
                    status text not null default 'active',
                    valid_from text,
                    valid_to text,
                    value_json text,
                    version integer not null default 1
                );

                create table if not exists memory_evidence (
                    memory_id text not null,
                    event_id text not null,
                    weight real not null default 1.0,
                    primary key (memory_id, event_id),
                    foreign key (memory_id) references canonical_memories(memory_id) on delete cascade,
                    foreign key (event_id) references raw_events(event_id) on delete cascade
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
        extracted: dict[str, Any] | None = None,
        main_class: str | None = None,
        sub_class: str | None = None,
        candidate_type: str | None = None,
        confidence: float = 0.0,
        source_start_idx: int | None = None,
        source_end_idx: int | None = None,
    ) -> None:
        self.initialize()
        if main_class is not None:
            _validate_main_class(main_class)
        with self.connect() as conn:
            conn.execute(
                """
                insert into raw_events(
                    event_id, ts, session_key, history_text, plain_text,
                    extracted_json, main_class, sub_class, candidate_type,
                    confidence, source_start_idx, source_end_idx
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ts,
                    session_key,
                    history_text,
                    plain_text,
                    json.dumps(extracted, ensure_ascii=False) if extracted is not None else None,
                    main_class,
                    sub_class,
                    candidate_type,
                    confidence,
                    source_start_idx,
                    source_end_idx,
                ),
            )

    def upsert_canonical_memory(
        self,
        *,
        memory_id: str,
        main_class: str,
        text: str,
        sub_class: str = "",
        confidence: float = 0.0,
        priority: float = 0.0,
        status: str = "active",
        valid_from: str | None = None,
        valid_to: str | None = None,
        value: dict[str, Any] | None = None,
        version: int = 1,
    ) -> None:
        self.initialize()
        _validate_main_class(main_class)
        with self.connect() as conn:
            conn.execute(
                """
                insert into canonical_memories(
                    memory_id, main_class, sub_class, text, confidence,
                    priority, status, valid_from, valid_to, value_json, version
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(memory_id) do update set
                    main_class=excluded.main_class,
                    sub_class=excluded.sub_class,
                    text=excluded.text,
                    confidence=excluded.confidence,
                    priority=excluded.priority,
                    status=excluded.status,
                    valid_from=excluded.valid_from,
                    valid_to=excluded.valid_to,
                    value_json=excluded.value_json,
                    version=excluded.version
                """,
                (
                    memory_id,
                    main_class,
                    sub_class,
                    text,
                    confidence,
                    priority,
                    status,
                    valid_from,
                    valid_to,
                    json.dumps(value, ensure_ascii=False) if value is not None else None,
                    version,
                ),
            )

    def add_evidence_link(self, *, memory_id: str, event_id: str, weight: float = 1.0) -> None:
        self.initialize()
        with self.connect() as conn:
            conn.execute(
                """
                insert into memory_evidence(memory_id, event_id, weight)
                values (?, ?, ?)
                on conflict(memory_id, event_id) do update set weight=excluded.weight
                """,
                (memory_id, event_id, weight),
            )

    def list_raw_events(self) -> list[dict[str, Any]]:
        self.initialize()
        with self.connect() as conn:
            rows = conn.execute(
                """
                select event_id, ts, session_key, history_text, plain_text,
                       extracted_json, main_class, sub_class, candidate_type,
                       confidence, source_start_idx, source_end_idx
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
                select memory_id, main_class, sub_class, text, confidence,
                       priority, status, valid_from, valid_to, value_json, version
                from canonical_memories
                order by main_class asc, sub_class asc, memory_id asc
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_evidence_links(self) -> list[dict[str, Any]]:
        self.initialize()
        with self.connect() as conn:
            rows = conn.execute(
                """
                select memory_id, event_id, weight
                from memory_evidence
                order by memory_id asc, event_id asc
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
        *,
        event_id: str | None = None,
    ) -> None:
        """Replace the canonical snapshot with a new set of canonical memories."""
        self.initialize()
        with self.connect() as conn:
            conn.execute("delete from canonical_memories")
            for memory in memories:
                main_class = str(memory["main_class"])
                _validate_main_class(main_class)
                sub_class = str(memory.get("sub_class") or "")
                text = str(memory.get("text") or "")
                conn.execute(
                    """
                    insert into canonical_memories(
                        memory_id, main_class, sub_class, text, confidence,
                        priority, status, valid_from, valid_to, value_json, version
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(memory.get("memory_id") or _canonical_memory_id(main_class, sub_class, text)),
                        main_class,
                        sub_class,
                        text,
                        float(memory.get("confidence", 0.0) or 0.0),
                        float(memory.get("priority", 0.0) or 0.0),
                        str(memory.get("status") or "active"),
                        memory.get("valid_from"),
                        memory.get("valid_to"),
                        json.dumps(memory.get("value"), ensure_ascii=False) if memory.get("value") is not None else None,
                        int(memory.get("version", 1) or 1),
                    ),
                )
                if event_id is not None:
                    conn.execute(
                        """
                        insert into memory_evidence(memory_id, event_id, weight)
                        values (?, ?, ?)
                        """,
                        (
                            str(memory.get("memory_id") or _canonical_memory_id(main_class, sub_class, text)),
                            event_id,
                            1.0,
                        ),
                    )

