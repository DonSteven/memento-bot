"""Application service for the single structured memory path."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from loguru import logger

from nanobot.agent.memory_db import MemoryContext, MemoryDatabase, MemorySnapshot, MemoryWriteResult
from nanobot.agent.memory_pipeline import (
    MemoryExtractionError,
    StructuredMemoryPipeline,
    format_messages,
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
        self._write_lock = asyncio.Lock()

    async def prepare_context(self, query: str, retrieval_budget: int = 5) -> MemoryContext:
        core_items = self.database.read_core_memories()
        retrieved_items = self.database.query_dynamic_memories(query, limit=retrieval_budget)
        core_ids = {item.memory_id for item in core_items}
        return MemoryContext(
            core_items=core_items,
            retrieved_items=tuple(item for item in retrieved_items if item.memory_id not in core_ids),
        )

    def sync_markdown(self) -> None:
        """Publish database views. Importing manual edits is intentionally deferred to P2."""
        self.database.write_views()

    async def consolidate(
        self,
        messages: list[dict[str, object]],
        *,
        session_key: str,
    ) -> MemoryWriteResult:
        if not messages:
            revision = self.database.read_snapshot().revision
            return MemoryWriteResult(True, True, revision)
        async with self._write_lock:
            current = self.database.read_snapshot()
            try:
                snapshot, history_entry = await self.pipeline.extract_snapshot(messages, current)
            except Exception as exc:
                if not isinstance(exc, MemoryExtractionError):
                    logger.exception("Structured memory extraction failed")
                else:
                    logger.warning("Structured memory extraction failed: {}", exc)
                return MemoryWriteResult(False, False, current.revision, str(exc))
            return self._commit(
                snapshot,
                expected_revision=current.revision,
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
            current = self.database.read_snapshot()
            now = datetime.now()
            plain_text = format_messages(messages)
            history_entry = (
                f"[{now.strftime('%Y-%m-%d %H:%M')}] [RAW] {len(messages)} messages\n"
                f"{plain_text}"
            )
            return self._commit(
                current,
                expected_revision=current.revision,
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
        session_key: str,
        history_text: str,
        plain_text: str,
        candidate_type: str,
    ) -> MemoryWriteResult:
        now = datetime.now().isoformat(timespec="seconds")
        event_id = f"{session_key}:{uuid4().hex}"
        try:
            revision = self.database.commit_snapshot(
                snapshot,
                expected_revision=expected_revision,
                event_id=event_id,
                ts=now,
                session_key=session_key,
                history_text=history_text,
                plain_text=plain_text,
                candidate_type=candidate_type,
            )
        except Exception as exc:
            logger.exception("Structured memory database commit failed")
            return MemoryWriteResult(False, False, expected_revision, str(exc))
        try:
            self.sync_markdown()
        except Exception as exc:
            logger.exception("Memory database committed at revision {}, but view export failed", revision)
            return MemoryWriteResult(True, False, revision, str(exc))
        return MemoryWriteResult(True, True, revision)
