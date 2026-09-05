"""Session-boundary and token-triggered memory consolidation policy."""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.memory_db import MemoryContext, MemoryWriteResult
from nanobot.agent.memory_service import MemoryService
from nanobot.utils.helpers import estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


class MemoryConsolidator:
    """Own consolidation triggers, session locks, and archive checkpoints."""

    _MAX_CONSOLIDATION_ROUNDS = 5
    _MAX_EXTRACTION_FAILURES = 3
    _SAFETY_BUFFER = 1024

    def __init__(
        self,
        memory_service: MemoryService,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ) -> None:
        self.memory_service = memory_service
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(
        self,
        messages: list[dict[str, object]],
        *,
        session_key: str = "memory",
    ) -> MemoryWriteResult:
        return await self.memory_service.consolidate(messages, session_key=session_key)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None
        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)
        return last_boundary

    def estimate_session_prompt_tokens(
        self, session: Session, memory_context: MemoryContext, *, current_message: str = "",
    ) -> tuple[int, str]:
        """Estimate using prepared memory without querying or extracting memory."""
        history = session.get_history(max_messages=0)
        channel, chat_id = session.key.split(":", 1) if ":" in session.key else (None, None)
        probe_messages = self._build_messages(
            history=history,
            current_message=current_message,
            channel=channel,
            chat_id=chat_id,
            memory_context=memory_context,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(
        self,
        messages: list[dict[str, object]],
        *,
        session_key: str = "memory",
    ) -> MemoryWriteResult:
        if not messages:
            revision = self.memory_service.database.read_snapshot().revision
            return MemoryWriteResult(True, True, revision)
        last_result: MemoryWriteResult | None = None
        for _ in range(self._MAX_EXTRACTION_FAILURES):
            last_result = await self.consolidate_messages(messages, session_key=session_key)
            if last_result.database_committed:
                return last_result
            if not last_result.retryable:
                return last_result
        logger.warning("Memory extraction repeatedly failed; persisting raw archive")
        raw_result = await self.memory_service.archive_raw(messages, session_key=session_key)
        return raw_result if raw_result.database_committed else last_result or raw_result

    async def maybe_consolidate_by_tokens(
        self, session: Session, memory_context: MemoryContext, *, query: str,
        current_message: str = "",
    ) -> MemoryContext:
        """Return prepared memory, refreshing it only after a committed archive."""
        if not session.messages or self.context_window_tokens <= 0:
            return memory_context
        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated, source = self.estimate_session_prompt_tokens(
                session, memory_context, current_message=current_message,
            )
            if estimated <= 0 or estimated < budget:
                return memory_context
            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return memory_context
                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug("Token consolidation: no safe boundary for {} (round {})", session.key, round_num)
                    return memory_context
                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return memory_context
                result = await self.archive_messages(chunk, session_key=session.key)
                if not result.database_committed:
                    logger.error("Memory archive failed for {}: {}", session.key, result.error)
                    return memory_context
                session.last_consolidated = end_idx
                self.sessions.save(session)
                if not result.view_exported:
                    logger.warning("Memory archive committed but view export is pending: {}", result.error)
                memory_context = await self.memory_service.prepare_context(query)
                estimated, source = self.estimate_session_prompt_tokens(
                    session, memory_context, current_message=current_message,
                )
                if estimated <= 0:
                    return memory_context
            return memory_context
