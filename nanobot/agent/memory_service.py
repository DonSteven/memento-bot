"""Application service for the single structured memory path."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from loguru import logger

from nanobot.agent.memory_db import (
    MemoryContext,
    MemoryDatabase,
    MemoryRevisionConflictError,
    MemorySnapshot,
    MemoryWriteResult,
)
from nanobot.agent.memory_pipeline import (
    MemoryExtractionError,
    StructuredMemoryPipeline,
    format_messages,
)
from nanobot.agent.memory_sync import (
    MarkdownValidationError,
    MemoryConflictError,
    MemorySynchronizer,
    MemorySyncResult,
    render_memory_markdown,
)
from nanobot.providers.base import LLMProvider


class MemoryService:
    """Coordinate extraction, transactional persistence, export, and retrieval."""

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        *,
        database: MemoryDatabase | None = None,
        pipeline: StructuredMemoryPipeline | None = None,
    ) -> None:
        self.database = database or MemoryDatabase(workspace)
        self.database.initialize()
        self.pipeline = pipeline or StructuredMemoryPipeline(provider, model)
        self.synchronizer = MemorySynchronizer(self.database)
        self._write_lock = asyncio.Lock()
        if self.synchronizer.read_state().pending_revision is not None:
            self.synchronizer.publish_pending()

    async def prepare_context(self, query: str, retrieval_budget: int = 5) -> MemoryContext:
        async with self._write_lock:
            self.synchronizer.sync()
            core_items = self.database.read_core_memories()
            retrieved_items = self.database.query_dynamic_memories(query, limit=retrieval_budget)
            core_ids = {item.memory_id for item in core_items}
            return MemoryContext(
                core_items=core_items,
                retrieved_items=tuple(
                    item for item in retrieved_items if item.memory_id not in core_ids
                ),
            )

    async def sync_markdown(self) -> MemorySyncResult:
        async with self._write_lock:
            return self.synchronizer.sync()

    async def consolidate(
        self,
        messages: list[dict[str, object]],
        *,
        session_key: str,
    ) -> MemoryWriteResult:
        if not messages:
            try:
                sync_result = await self.sync_markdown()
            except (MarkdownValidationError, MemoryConflictError) as exc:
                revision = self.database.read_snapshot().revision
                return MemoryWriteResult(False, False, revision, str(exc), retryable=False)
            return MemoryWriteResult(True, True, sync_result.revision)
        async with self._write_lock:
            try:
                self.synchronizer.sync()
            except (MarkdownValidationError, MemoryConflictError) as exc:
                revision = self.database.read_snapshot().revision
                return MemoryWriteResult(False, False, revision, str(exc), retryable=False)
            current = self.database.read_snapshot()
            expected_file_text = self.synchronizer.read_memory_file()
            try:
                snapshot, history_entry = await self.pipeline.extract_snapshot(messages, current)
            except Exception as exc:
                if not isinstance(exc, MemoryExtractionError):
                    logger.exception("Structured memory extraction failed")
                else:
                    logger.warning("Structured memory extraction failed: {}", exc)
                return MemoryWriteResult(False, False, current.revision, str(exc))

            latest = self.database.read_snapshot()
            latest_file_text = self.synchronizer.read_memory_file()
            if latest.revision != current.revision or latest_file_text != expected_file_text:
                try:
                    self.synchronizer.sync()
                except (MarkdownValidationError, MemoryConflictError) as exc:
                    return MemoryWriteResult(
                        False,
                        False,
                        self.database.read_snapshot().revision,
                        str(exc),
                        retryable=False,
                    )
                return MemoryWriteResult(
                    False,
                    False,
                    self.database.read_snapshot().revision,
                    "Memory changed while extraction was running; the stale extraction result was discarded",
                    retryable=False,
                )
            return self._commit(
                snapshot,
                expected_revision=current.revision,
                publish_expected_text=expected_file_text,
                session_key=session_key,
                history_text=history_entry,
                plain_text=format_messages(messages),
                candidate_type="snapshot",
            )

    async def archive_raw(
        self,
        messages: list[dict[str, object]],
        *,
        session_key: str,
    ) -> MemoryWriteResult:
        """Persist a raw history event after repeated extraction failures."""
        async with self._write_lock:
            try:
                self.synchronizer.sync()
            except (MarkdownValidationError, MemoryConflictError) as exc:
                revision = self.database.read_snapshot().revision
                return MemoryWriteResult(False, False, revision, str(exc), retryable=False)
            current = self.database.read_snapshot()
            expected_file_text = self.synchronizer.read_memory_file()
            now = datetime.now()
            plain_text = format_messages(messages)
            history_entry = (
                f"[{now.strftime('%Y-%m-%d %H:%M')}] [RAW] {len(messages)} messages\n{plain_text}"
            )
            return self._commit(
                current,
                expected_revision=current.revision,
                publish_expected_text=expected_file_text,
                session_key=session_key,
                history_text=history_entry,
                plain_text=plain_text,
                candidate_type="raw_archive",
            )

    def _commit(
        self,
        snapshot: MemorySnapshot,
        *,
        expected_revision: int,
        publish_expected_text: str | None,
        session_key: str,
        history_text: str,
        plain_text: str,
        candidate_type: str,
    ) -> MemoryWriteResult:
        now = datetime.now().isoformat(timespec="seconds")
        event_id = f"{session_key}:{uuid4().hex}"
        try:
            target_text = render_memory_markdown(snapshot.memories)
            revision = self.database.commit_snapshot(
                snapshot,
                expected_revision=expected_revision,
                event_id=event_id,
                ts=now,
                session_key=session_key,
                history_text=history_text,
                plain_text=plain_text,
                candidate_type=candidate_type,
                publish_expected_text=publish_expected_text,
                publish_target_text=target_text,
                stage_publish=True,
            )
        except Exception as exc:
            logger.exception("Structured memory database commit failed")
            retryable = not isinstance(
                exc, (MemoryConflictError, MarkdownValidationError, MemoryRevisionConflictError)
            )
            return MemoryWriteResult(False, False, expected_revision, str(exc), retryable=retryable)
        try:
            self.synchronizer.publish_pending()
        except Exception as exc:
            logger.exception(
                "Memory database committed at revision {}, but view export failed", revision
            )
            return MemoryWriteResult(True, False, revision, str(exc))
        return MemoryWriteResult(True, True, revision)
