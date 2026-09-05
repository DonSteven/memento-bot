"""Application service for the single structured memory path."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import tiktoken
from loguru import logger

from nanobot.agent.memory_db import (
    DYNAMIC_MEMORY_CLASSES,
    DynamicMemoryHit,
    MemoryContext,
    MemoryDatabase,
    MemoryRecord,
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
from nanobot.agent.retrieval import DashScopeEmbeddingBackend, EmbeddingBackend, passes_lexical_gate
from nanobot.config.schema import MemoryConfig
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
        config: MemoryConfig | None = None,
        api_key: str | None = None,
        embedder: EmbeddingBackend | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        embedding_config = self.config.embedding
        self.embedder = embedder or DashScopeEmbeddingBackend(
            api_key=api_key or "",
            api_base=embedding_config.api_base,
            model=embedding_config.model,
            dimension=embedding_config.dimensions,
        )
        self.database = database or MemoryDatabase(workspace)
        self.database.initialize(
            self.embedder.dimension,
            embedding_provider=embedding_config.provider,
            embedding_model=embedding_config.model,
        )
        self.pipeline = pipeline or StructuredMemoryPipeline(provider, model)
        self.synchronizer = MemorySynchronizer(self.database)
        self._write_lock = asyncio.Lock()
        if self.synchronizer.read_state().pending_revision is not None:
            self.synchronizer.publish_pending()

    async def prepare_context(
        self, query: str, retrieval_budget: int | None = None
    ) -> MemoryContext:
        async with self._write_lock:
            await self.synchronizer.sync(self._embed_records)
            core_items = self.database.read_core_memories()
            hits = await self.search_dynamic(query, limit=retrieval_budget)
            retrieved_items = self._apply_token_budget(
                (hit.record for hit in hits), self.config.dynamic_token_budget,
                core_items=core_items,
            )
            core_ids = {item.memory_id for item in core_items}
            return MemoryContext(
                core_items=core_items,
                retrieved_items=tuple(
                    item for item in retrieved_items if item.memory_id not in core_ids
                ),
            )

    async def sync_markdown(self) -> MemorySyncResult:
        async with self._write_lock:
            return await self.synchronizer.sync(self._embed_records)

    async def search_dynamic(
        self, query: str, *, limit: int | None = None
    ) -> tuple[DynamicMemoryHit, ...]:
        normalized = " ".join(query.split())
        requested_limit = self.config.dynamic_top_k if limit is None else limit
        resolved_limit = min(max(requested_limit, 0), self.config.dynamic_top_k)
        if not normalized or resolved_limit <= 0:
            return ()
        # Avoid a paid query call when there is no dynamic memory to search.
        if not any(
            item.main_class in DYNAMIC_MEMORY_CLASSES
            for item in self.database.read_snapshot().memories
        ):
            return ()
        query_vector = await self.embedder.embed_query(normalized)
        fts_hits = self.database.search_dynamic_fts(normalized, self.config.fts_recall_limit)
        vector_hits = [
            item
            for item in self.database.search_dynamic_vector(
                query_vector, self.config.vector_recall_limit
            )
            if float(item["vector_similarity"]) >= self.config.vector_similarity_threshold
        ]
        eligible_ids = {
            item["memory_id"] for item in fts_hits
            if passes_lexical_gate(normalized, f"{item['sub_class']} {item['text']}")
        } | {item["memory_id"] for item in vector_hits}
        fused = self.database.fuse_dynamic(
            {"fts": fts_hits, "vector": vector_hits}, len(fts_hits) + len(vector_hits)
        )
        fused = [item for item in fused if item["memory_id"] in eligible_ids][:resolved_limit]
        return tuple(
            DynamicMemoryHit(
                record=MemoryRecord(
                    item["memory_id"], item["main_class"], item["sub_class"], item["text"]
                ),
                sources=tuple(item["sources"]),
                rrf_score=float(item["rrf_score"]),
                fts_rank=item.get("fts_rank"),
                vector_rank=item.get("vector_rank"),
                vector_similarity=item.get("vector_similarity"),
            )
            for item in fused
        )

    async def _embed_records(self, records) -> dict[str, list[float]]:
        dynamic = tuple(item for item in records if item.main_class in DYNAMIC_MEMORY_CLASSES)
        if not dynamic:
            return {}
        vectors = await self.embedder.embed_documents([item.text for item in dynamic])
        if len(vectors) != len(dynamic):
            raise RuntimeError("Embedding backend returned an unexpected number of memory vectors")
        return {item.memory_id: vector for item, vector in zip(dynamic, vectors)}

    @staticmethod
    def _apply_token_budget(
        records: Iterable[MemoryRecord], budget: int,
        *, core_items: tuple[MemoryRecord, ...] = (),
    ) -> tuple[MemoryRecord, ...]:
        if budget <= 0:
            return ()
        encoding = tiktoken.get_encoding("cl100k_base")
        selected: list[MemoryRecord] = []
        core_block = MemoryContext(core_items=core_items).render_block()
        for record in records:
            block = MemoryContext(core_items, (*selected, record)).render_block()
            dynamic_block = block[len(core_block):]
            if len(encoding.encode(dynamic_block, disallowed_special=())) > budget:
                continue
            selected.append(record)
        return tuple(selected)

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
                await self.synchronizer.sync(self._embed_records)
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
                    await self.synchronizer.sync(self._embed_records)
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
            return await self._commit(
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
                await self.synchronizer.sync(self._embed_records)
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
            return await self._commit(
                current,
                expected_revision=current.revision,
                publish_expected_text=expected_file_text,
                session_key=session_key,
                history_text=history_entry,
                plain_text=plain_text,
                candidate_type="raw_archive",
            )

    async def _commit(
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
            current = self.database.read_snapshot()
            existing_ids = {item.memory_id for item in current.memories}
            changed_dynamic = tuple(
                item
                for item in snapshot.memories
                if item.main_class in DYNAMIC_MEMORY_CLASSES and item.memory_id not in existing_ids
            )
            dynamic_embeddings = await self._embed_records(changed_dynamic)
            latest = self.database.read_snapshot()
            latest_file_text = self.synchronizer.read_memory_file()
            if latest.revision != expected_revision or latest_file_text != publish_expected_text:
                raise MemoryRevisionConflictError("Memory changed while embedding was running")
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
                dynamic_embeddings=dynamic_embeddings,
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
