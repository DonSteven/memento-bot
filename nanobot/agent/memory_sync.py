"""Strict Markdown synchronization for the SQLite-backed memory snapshot."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable
from uuid import uuid4

from nanobot.agent.memory_db import (
    MEMORY_CLASSES,
    MemoryDatabase,
    MemoryRecord,
    MemorySnapshot,
    validate_memory_fields,
)

MEMORY_SECTIONS = (
    ("personal_profile", "Personal Profile", "(Stable background information about the user)"),
    ("preferences", "Preferences", "(How the user prefers to communicate and collaborate)"),
    ("constraints", "Constraints", "(Rules, boundaries, and requirements that must be respected)"),
    ("projects", "Projects", "(Information about ongoing learning and work projects)"),
    ("daily_life", "Daily Life", "(Daily routines, interests, and hobbies)"),
    (
        "plans_commitments",
        "Plans and Commitments",
        "(Future plans, commitments, deadlines, and to-dos)",
    ),
)
_TITLE_TO_CLASS = {title: main_class for main_class, title, _ in MEMORY_SECTIONS}
_PLACEHOLDER_BY_CLASS = {main_class: placeholder for main_class, _, placeholder in MEMORY_SECTIONS}
_DOCUMENT_LINES = {
    "# Long-term Memory",
    "This file stores important information that should persist across sessions.",
    "---",
    "*This file is automatically updated by nanobot when important information should be remembered.*",
}


class MarkdownValidationError(ValueError):
    """Raised when MEMORY.md is not a complete, valid editable memory view."""

    def __init__(self, line_number: int, message: str) -> None:
        self.line_number = line_number
        super().__init__(f"MEMORY.md line {line_number}: {message}")


class MemoryConflictError(RuntimeError):
    """Raised when database and file state cannot be reconciled without data loss."""


class MemorySyncStatus(StrEnum):
    UNCHANGED = "unchanged"
    IMPORTED = "imported"
    PUBLISHED = "published"
    RECOVERED = "recovered"
    NORMALIZED = "normalized"


@dataclass(frozen=True, slots=True)
class MemorySyncState:
    published_revision: int | None
    published_text: str | None
    pending_revision: int | None
    pending_expected_text: str | None
    pending_target_text: str | None


@dataclass(frozen=True, slots=True)
class MemoryChangeSet:
    added: tuple[MemoryRecord, ...] = ()
    removed: tuple[MemoryRecord, ...] = ()
    unchanged: tuple[MemoryRecord, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


@dataclass(frozen=True, slots=True)
class MemorySyncResult:
    status: MemorySyncStatus
    revision: int
    changes: MemoryChangeSet = MemoryChangeSet()


def render_memory_markdown(memories: Iterable[MemoryRecord | dict[str, Any]]) -> str:
    grouped: dict[str, list[MemoryRecord]] = {key: [] for key in MEMORY_CLASSES}
    for value in memories:
        if isinstance(value, MemoryRecord):
            record = value
        else:
            main_class = str(value.get("main_class") or "").strip()
            sub_class = str(value.get("sub_class") or "").strip()
            text = str(value.get("text") or "").strip()
            generated = MemoryRecord.create(main_class, sub_class, text)
            record = MemoryRecord(
                str(value.get("memory_id") or generated.memory_id), main_class, sub_class, text
            )
        validate_memory_fields(record.main_class, record.sub_class, record.text)
        grouped[record.main_class].append(record)

    lines = [
        "# Long-term Memory",
        "",
        "This file stores important information that should persist across sessions.",
        "",
    ]
    for main_class, title, placeholder in MEMORY_SECTIONS:
        lines.extend([f"## {title}", ""])
        records = sorted(
            grouped[main_class],
            key=lambda item: (item.sub_class, item.memory_id, item.text),
        )
        if records:
            lines.extend(f"- {item.sub_class}: {item.text}" for item in records)
        else:
            lines.append(placeholder)
        lines.append("")
    lines.extend(
        [
            "---",
            "",
            "*This file is automatically updated by nanobot when important information should be remembered.*",
        ]
    )
    return "\n".join(lines)


def parse_memory_markdown(content: str) -> tuple[MemoryRecord, ...]:
    """Parse a complete canonical memory document and report the first invalid line."""
    if not content:
        raise MarkdownValidationError(1, "empty files are not valid memory snapshots")

    current_class: str | None = None
    seen_sections: list[str] = []
    placeholder_sections: set[str] = set()
    records: list[MemoryRecord] = []
    seen_records: set[tuple[str, str, str]] = set()

    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            title = line[3:].strip()
            main_class = _TITLE_TO_CLASS.get(title)
            if main_class is None:
                raise MarkdownValidationError(line_number, f"unknown memory category {title!r}")
            if main_class in seen_sections:
                raise MarkdownValidationError(line_number, f"duplicate memory category {title!r}")
            expected_index = len(seen_sections)
            if (
                expected_index >= len(MEMORY_SECTIONS)
                or MEMORY_SECTIONS[expected_index][0] != main_class
            ):
                raise MarkdownValidationError(
                    line_number, "memory categories must use the canonical order"
                )
            seen_sections.append(main_class)
            current_class = main_class
            continue
        if line in _DOCUMENT_LINES:
            continue
        if line.startswith("#"):
            raise MarkdownValidationError(line_number, "unsupported heading")
        if current_class is None:
            raise MarkdownValidationError(line_number, "content appears outside a memory category")
        if line == _PLACEHOLDER_BY_CLASS[current_class]:
            if any(record.main_class == current_class for record in records):
                raise MarkdownValidationError(
                    line_number, "an empty-category placeholder cannot follow entries"
                )
            placeholder_sections.add(current_class)
            continue
        if not line.startswith("- "):
            raise MarkdownValidationError(line_number, "expected '- sub_class: text'")
        if current_class in placeholder_sections:
            raise MarkdownValidationError(
                line_number, "entries cannot follow an empty-category placeholder"
            )
        payload = line[2:]
        if ": " not in payload:
            raise MarkdownValidationError(line_number, "expected '- sub_class: text'")
        sub_class, text = (part.strip() for part in payload.split(": ", 1))
        if not sub_class:
            raise MarkdownValidationError(line_number, "sub_class must not be empty")
        if not text:
            raise MarkdownValidationError(line_number, "memory text must not be empty")
        key = (current_class, sub_class, text)
        if key in seen_records:
            raise MarkdownValidationError(line_number, "duplicate memory entry")
        seen_records.add(key)
        records.append(MemoryRecord.create(*key))

    if tuple(seen_sections) != MEMORY_CLASSES:
        missing = [
            title for main_class, title, _ in MEMORY_SECTIONS if main_class not in seen_sections
        ]
        raise MarkdownValidationError(
            len(content.splitlines()) + 1, f"missing memory categories: {', '.join(missing)}"
        )
    return tuple(records)


def diff_memory_snapshots(
    current: MemorySnapshot,
    parsed_records: tuple[MemoryRecord, ...],
) -> tuple[MemorySnapshot, MemoryChangeSet]:
    """Compare category/subclass/text triples while retaining IDs for unchanged records."""
    existing = {(item.main_class, item.sub_class, item.text): item for item in current.memories}
    parsed = {(item.main_class, item.sub_class, item.text): item for item in parsed_records}
    unchanged = tuple(existing[key] for key in existing.keys() & parsed.keys())
    added = tuple(parsed[key] for key in parsed.keys() - existing.keys())
    removed = tuple(existing[key] for key in existing.keys() - parsed.keys())
    target_records = tuple(
        sorted(
            (*unchanged, *added), key=lambda item: (item.main_class, item.sub_class, item.memory_id)
        )
    )
    changes = MemoryChangeSet(
        added=tuple(sorted(added, key=lambda item: item.memory_id)),
        removed=tuple(sorted(removed, key=lambda item: item.memory_id)),
        unchanged=tuple(sorted(unchanged, key=lambda item: item.memory_id)),
    )
    return MemorySnapshot(current.revision, target_records), changes


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_checked(path: Path, *, expected: str | None, target: str) -> None:
    """Prepare the replacement, then compare the file again immediately before replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(target)
            stream.flush()
            os.fsync(stream.fileno())
        current = _read_optional(path)
        if current == target:
            temporary_path.unlink()
            return
        if current != expected:
            raise MemoryConflictError(
                "MEMORY.md changed immediately before publication; the file was preserved"
            )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class MemorySynchronizer:
    """Import manual edits and publish committed snapshots without silent overwrite."""

    def __init__(self, database: MemoryDatabase) -> None:
        self.database = database
        self.memory_file = database.memory_file
        self.history_file = database.history_file

    def read_state(self) -> MemorySyncState:
        raw = self.database.read_publish_state()
        return MemorySyncState(
            published_revision=raw["published_revision"],  # type: ignore[arg-type]
            published_text=raw["published_text"],  # type: ignore[arg-type]
            pending_revision=raw["pending_revision"],  # type: ignore[arg-type]
            pending_expected_text=raw["pending_expected_text"],  # type: ignore[arg-type]
            pending_target_text=raw["pending_target_text"],  # type: ignore[arg-type]
        )

    def read_memory_file(self) -> str | None:
        return _read_optional(self.memory_file)

    def recover_pending(self) -> MemorySyncResult | None:
        state = self.read_state()
        if state.pending_revision is None:
            return None
        target = state.pending_target_text
        if target is None:
            raise RuntimeError("Pending memory publication has no target text")
        current = self.read_memory_file()
        if current == state.pending_expected_text:
            _atomic_write_checked(
                self.memory_file,
                expected=state.pending_expected_text,
                target=target,
            )
        elif current != target:
            raise MemoryConflictError(
                "MEMORY.md changed while a committed database snapshot was awaiting publication; "
                "the file was preserved and the database target remains pending"
            )
        # Keep the intent pending until both views are durable. This also recovers
        # a history export failure after MEMORY.md has already been replaced.
        self.publish_history()
        self.database.confirm_memory_publish(revision=state.pending_revision, target_text=target)
        return MemorySyncResult(MemorySyncStatus.RECOVERED, state.pending_revision)

    def publish_pending(self) -> MemorySyncResult:
        recovered = self.recover_pending()
        if recovered is None:
            state = self.read_state()
            revision = state.published_revision
            if revision is None:
                revision = self.database.read_snapshot().revision
            recovered = MemorySyncResult(MemorySyncStatus.UNCHANGED, revision)
            if not self.history_file.exists():
                self.publish_history()
        return recovered

    def publish_history(self) -> None:
        _atomic_write(self.history_file, self.database.render_history_view())

    async def sync(
        self,
        embed_records: Callable[[tuple[MemoryRecord, ...]], Awaitable[dict[str, list[float]]]],
    ) -> MemorySyncResult:
        recovered = self.recover_pending()
        current = self.database.read_snapshot()
        target_text = render_memory_markdown(current.memories)
        state = self.read_state()
        file_text = self.read_memory_file()

        if state.published_revision is None:
            if file_text is None:
                self.database.stage_memory_publish(
                    revision=current.revision,
                    expected_text=None,
                    target_text=target_text,
                )
                self.publish_pending()
                return MemorySyncResult(MemorySyncStatus.PUBLISHED, current.revision)
            if file_text == "":
                raise MarkdownValidationError(1, "empty files are not valid memory snapshots")
            if file_text == target_text:
                self.publish_history()
                self.database.record_memory_publish_baseline(
                    revision=current.revision, text=file_text
                )
                return MemorySyncResult(MemorySyncStatus.UNCHANGED, current.revision)
            if current.revision != 0 or current.memories:
                raise MemoryConflictError(
                    "MEMORY.md has no synchronization baseline and differs from the database snapshot"
                )
            return await self._import_file(current, file_text, embed_records)

        if file_text is None:
            self.database.stage_memory_publish(
                revision=current.revision,
                expected_text=None,
                target_text=target_text,
            )
            self.publish_pending()
            return MemorySyncResult(MemorySyncStatus.PUBLISHED, current.revision)
        if file_text == "":
            raise MarkdownValidationError(1, "empty files are not valid memory snapshots")

        if file_text == state.published_text:
            if current.revision != state.published_revision:
                self.database.stage_memory_publish(
                    revision=current.revision,
                    expected_text=file_text,
                    target_text=target_text,
                )
                self.publish_pending()
                return MemorySyncResult(MemorySyncStatus.PUBLISHED, current.revision)
            if not self.history_file.exists():
                self.publish_history()
            status = (
                MemorySyncStatus.RECOVERED if recovered is not None else MemorySyncStatus.UNCHANGED
            )
            return MemorySyncResult(status, current.revision)

        if current.revision != state.published_revision:
            raise MemoryConflictError(
                "Both MEMORY.md and the database changed from the last published revision"
            )
        return await self._import_file(current, file_text, embed_records)

    async def _import_file(
        self,
        current: MemorySnapshot,
        file_text: str,
        embed_records: Callable[[tuple[MemoryRecord, ...]], Awaitable[dict[str, list[float]]]],
    ) -> MemorySyncResult:
        parsed = parse_memory_markdown(file_text)
        target, changes = diff_memory_snapshots(current, parsed)
        normalized = render_memory_markdown(target.memories)
        if not changes.changed:
            if file_text != normalized:
                self.database.stage_memory_publish(
                    revision=current.revision,
                    expected_text=file_text,
                    target_text=normalized,
                )
                self.publish_pending()
                return MemorySyncResult(MemorySyncStatus.NORMALIZED, current.revision, changes)
            self.publish_history()
            self.database.record_memory_publish_baseline(revision=current.revision, text=file_text)
            return MemorySyncResult(MemorySyncStatus.UNCHANGED, current.revision, changes)

        now = datetime.now()
        history_text = (
            f"[{now.strftime('%Y-%m-%d %H:%M')}] Imported manual MEMORY.md edit "
            f"(+{len(changes.added)} -{len(changes.removed)})"
        )
        dynamic_embeddings = await embed_records(changes.added)
        latest = self.database.read_snapshot()
        if latest.revision != current.revision or self.read_memory_file() != file_text:
            raise MemoryConflictError(
                "Memory changed while manual edits were being embedded; no changes were committed"
            )
        revision = self.database.commit_snapshot(
            target,
            expected_revision=current.revision,
            event_id=f"markdown:{uuid4().hex}",
            ts=now.isoformat(timespec="seconds"),
            session_key="memory:markdown",
            history_text=history_text,
            plain_text=file_text,
            candidate_type="markdown_edit",
            publish_expected_text=file_text,
            publish_target_text=normalized,
            stage_publish=True,
            dynamic_embeddings=dynamic_embeddings,
        )
        self.publish_pending()
        return MemorySyncResult(MemorySyncStatus.IMPORTED, revision, changes)
