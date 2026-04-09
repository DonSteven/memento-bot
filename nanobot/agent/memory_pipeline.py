"""Structured memory pipeline for the DB-backed v2 memory path."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.memory_db import MemoryDatabase, _canonical_memory_id

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


_SAVE_MEMORY_STRUCTURED_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory_structured",
            "description": "Save the memory consolidation result to persistent structured storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "canonical_memories": {
                        "type": "array",
                        "description": "Full updated long-term memory as structured canonical items. "
                        "Include all existing facts plus new ones. Remove outdated facts by omitting them. "
                        "Every memory must include an explicit non-empty subclass; reuse an existing subclass "
                        "or create a new one when needed.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "main_class": {
                                    "type": "string",
                                    "enum": [
                                        "personal_profile",
                                        "preferences",
                                        "constraints",
                                        "projects",
                                        "daily_life",
                                        "plans_commitments",
                                    ],
                                },
                                "sub_class": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["main_class", "sub_class", "text"],
                        },
                    },
                },
                "required": ["history_entry", "canonical_memories"],
            },
        },
    }
]

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)
_FTS_CORE_CLASSES = ("personal_profile", "preferences", "constraints")


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for storage and comparison."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_structured_args(args: Any) -> dict[str, Any] | None:
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


def _is_tool_choice_unsupported(content: str | None) -> bool:
    text = (content or "").lower()
    return any(marker in text for marker in _TOOL_CHOICE_ERROR_MARKERS)


class StructuredMemoryPipeline:
    """DB-backed structured memory consolidation for v2."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3
    _SESSION_KEY = "v2"
    _SNAPSHOT_CANDIDATE_TYPE = "v2_snapshot"
    _RAW_ARCHIVE_CANDIDATE_TYPE = "v2_raw_archive"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.db = MemoryDatabase(workspace)
        self._consecutive_failures = 0

    @staticmethod
    def _format_memory_snippets(items: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for item in items:
            main_class = str(item.get("main_class") or "")
            sub_class = str(item.get("sub_class") or "").strip()
            text = str(item.get("text") or "").strip()
            if not sub_class or not text:
                continue
            label = f"{main_class}/{sub_class}"
            lines.append(f"- [{label}] {text}")
        return lines

    @staticmethod
    def _memory_identity(item: dict[str, Any]) -> str:
        memory_id = str(item.get("memory_id") or "").strip()
        if memory_id:
            return memory_id
        return "|".join([
            str(item.get("main_class") or "").strip(),
            str(item.get("sub_class") or "").strip(),
            str(item.get("text") or "").strip(),
        ])

    def build_retrieval_context(self, query: str, limit: int = 5) -> str:
        """ 把始终应该带上的核心记忆 和 和当前 query 相关的检索记忆 组合成一段可直接塞进 prompt 的上下文字符串。"""
        if not self.db.db_path.exists():
            return ""

        core_items = [
            item
            for item in self.db.list_canonical_memories()
            if str(item.get("main_class") or "").strip() in _FTS_CORE_CLASSES
        ]
        retrieved_items: list[dict[str, Any]] = []
        seen = {self._memory_identity(item) for item in core_items}
        if query.strip():
            for item in self.db.query_canonical_memories(query, limit=limit):
                identity = self._memory_identity(item)
                if identity in seen:
                    continue
                seen.add(identity)
                retrieved_items.append(item)

        if not core_items and not retrieved_items:
            return ""

        lines: list[str] = []
        core_lines = self._format_memory_snippets(core_items)
        if core_lines:
            lines.extend(["## Core Memory", *core_lines])
        retrieved_lines = self._format_memory_snippets(retrieved_items)
        if retrieved_lines:
            if lines:
                lines.append("")
            lines.extend(["## Retrieved Memory", *retrieved_lines])
        return "\n".join(lines)

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    def _build_consolidation_memory_view(self) -> str:
        if self.db.memory_file.exists():
            return self.db.memory_file.read_text(encoding="utf-8")
        if self.db.db_path.exists():
            return self.db.render_memory_view()
        return "(empty)"

    @staticmethod
    def _normalize_snapshot_memories(items: list[Any]) -> list[dict[str, Any]]:
        """Normalize the structured snapshot and ensure each memory gets a unique id."""
        normalized: dict[str, dict[str, Any]] = {}

        for item in items:
            if not isinstance(item, dict):
                return []

            main_class = str(item.get("main_class") or "").strip()
            sub_class = _ensure_text(item.get("sub_class", "")).strip()
            text = _ensure_text(item.get("text", "")).strip()
            if not main_class or not sub_class or not text:
                return []
            memory_id = _canonical_memory_id(main_class, sub_class, text)
            normalized[memory_id] = {
                "memory_id": memory_id,
                "main_class": main_class,
                "sub_class": sub_class,
                "text": text,
            }

        return list(normalized.values())

    def _fail_or_raw_archive(self, messages: list[dict[str, Any]]) -> bool:
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict[str, Any]]) -> None:
        self.db.initialize()
        ts = datetime.now()
        plain_text = self._format_messages(messages)
        history_entry = (
            f"[{ts.strftime('%Y-%m-%d %H:%M')}] [RAW] {len(messages)} messages\n"
            f"{plain_text}"
        )
        event_id = (
            f"{self._SESSION_KEY}_raw_"
            f"{hashlib.sha1(f'{history_entry}|{ts.isoformat()}'.encode('utf-8')).hexdigest()[:12]}"
        )
        self.db.insert_raw_event(
            event_id=event_id,
            ts=ts.isoformat(timespec="seconds"),
            session_key=self._SESSION_KEY,
            history_text=history_entry,
            plain_text=plain_text,
            candidate_type=self._RAW_ARCHIVE_CANDIDATE_TYPE,
        )
        self.db.write_views()
        logger.warning(
            "Structured memory consolidation degraded: raw-archived {} messages",
            len(messages),
        )

    async def ingest_chunk(
        self,
        messages: list[dict[str, Any]],
        provider: LLMProvider,
        model: str,
    ) -> bool:
        if not messages:
            return True

        prompt = f"""
                Process this conversation and call the save_memory_structured tool with your consolidation.
                
                ## Current Long-term Memory
                {self._build_consolidation_memory_view()}

                ## Conversation to Process
                {self._format_messages(messages)}
                """

        chat_messages = [
            {
                "role": "system",
                "content": "You are a memory consolidation agent. Call the save_memory_structured tool with your consolidation of the conversation.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory_structured"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_STRUCTURED_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                logger.warning("Forced tool_choice unsupported in v2 mode, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_STRUCTURED_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Structured memory consolidation: LLM did not call save_memory_structured "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_structured_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Structured memory consolidation: unexpected save_memory_structured arguments")
                return self._fail_or_raw_archive(messages)
            if "history_entry" not in args or "canonical_memories" not in args:
                logger.warning("Structured memory consolidation: payload missing required fields")
                return self._fail_or_raw_archive(messages)

            history_entry = _ensure_text(args["history_entry"]).strip()
            canonical_memories = args["canonical_memories"]
            if not history_entry or not isinstance(canonical_memories, list):
                logger.warning("Structured memory consolidation: payload contains invalid required fields")
                return self._fail_or_raw_archive(messages)

            normalized_snapshot = self._normalize_snapshot_memories(canonical_memories)
            if canonical_memories and not normalized_snapshot:
                logger.warning("Structured memory consolidation: no valid canonical memories after normalization")
                return self._fail_or_raw_archive(messages)

            now = datetime.now().isoformat(timespec="seconds")
            event_id = (
                f"{self._SESSION_KEY}_evt_"
                f"{hashlib.sha1(f'{history_entry}|{now}'.encode('utf-8')).hexdigest()[:12]}"
            )
            self.db.write_structured_snapshot(
                event_id=event_id,
                ts=now,
                session_key=self._SESSION_KEY,
                history_text=history_entry,
                plain_text=self._format_messages(messages),
                candidate_type=self._SNAPSHOT_CANDIDATE_TYPE,
                memories=normalized_snapshot,
            )
            self._consecutive_failures = 0
            logger.info("Structured memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Structured memory consolidation failed")
            return self._fail_or_raw_archive(messages)
