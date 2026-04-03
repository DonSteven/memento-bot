"""Shadow memory pipeline for structured sidecar snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.memory_db import MemoryDatabase, _canonical_memory_id
from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


_SAVE_MEMORY_SHADOW_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory_shadow",
            "description": "Save the shadow memory consolidation result to structured storage.",
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
                        "Include all existing facts plus new ones. Remove outdated facts by omitting them.",
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
                                "confidence": {"type": "number"},
                            },
                            "required": ["main_class", "text"],
                        },
                    },
                },
                "required": ["history_entry", "canonical_memories"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for storage and comparison."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_shadow_args(args: Any) -> dict[str, Any] | None:
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    text = (content or "").lower()
    return any(marker in text for marker in _TOOL_CHOICE_ERROR_MARKERS)


class ShadowMemoryPipeline:
    """Best-effort sidecar pipeline that writes structured shadow memory snapshots."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.shadow_dir = ensure_dir(workspace / "memory" / "shadow")
        self.db = MemoryDatabase(
            workspace,
            storage_dir=self.shadow_dir,
            view_dir=self.shadow_dir,
        )
        self.debug_file = self.shadow_dir / "last_payload.json"
        self._consecutive_failures = 0

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

    def _write_debug_payload(self, payload: dict[str, Any]) -> None:
        self.debug_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize_snapshot_memories(items: list[Any]) -> list[dict[str, Any]]:
        """Normalize the structured snapshot and ensure each memory gets a stable unique id."""
        normalized: dict[str, dict[str, Any]] = {}

        for item in items:
            if not isinstance(item, dict):
                continue

            main_class = str(item.get("main_class") or "").strip()
            text = _ensure_text(item.get("text", "")).strip()
            if not main_class or not text:
                continue

            sub_class = _ensure_text(item.get("sub_class", "")).strip()
            memory_id = _canonical_memory_id(main_class, sub_class, text)
            normalized[memory_id] = {
                "memory_id": memory_id,
                "main_class": main_class,
                "sub_class": sub_class,
                "text": text,
                "confidence": float(item.get("confidence", 0.0) or 0.0),
            }

        return list(normalized.values())

    def _fail_or_raw_archive(self, messages: list[dict[str, Any]]) -> bool:
        """Mirror legacy failure handling for the sidecar path."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict[str, Any]]) -> None:
        """Fallback: persist the raw chunk into shadow raw_events without updating snapshot memory."""
        self.db.initialize()
        ts = datetime.now()
        plain_text = self._format_messages(messages)
        history_entry = (
            f"[{ts.strftime('%Y-%m-%d %H:%M')}] [RAW] {len(messages)} messages\n"
            f"{plain_text}"
        )
        event_id = f"shadow_raw_{hashlib.sha1(f'{history_entry}|{ts.isoformat()}'.encode('utf-8')).hexdigest()[:12]}"
        self.db.insert_raw_event(
            event_id=event_id,
            ts=ts.isoformat(timespec="seconds"),
            session_key="shadow",
            history_text=history_entry,
            plain_text=plain_text,
            extracted={"raw_archive": True, "message_count": len(messages)},
            candidate_type="shadow_raw_archive",
        )
        self.db.write_views()
        logger.warning(
            "Shadow memory consolidation degraded: raw-archived {} messages",
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

        current_shadow_memory = self.db.render_memory_view() if self.db.db_path.exists() else "(empty)"
        prompt = f"""
                Process this conversation and call the save_memory_shadow tool with your consolidation.

                ## Current Long-term Memory
                {current_shadow_memory}

                ## Conversation to Process
                {self._format_messages(messages)}
                """

        chat_messages = [
            {
                "role": "system",
                "content": "You are a memory consolidation agent. Call the save_memory_shadow tool with your consolidation of the conversation.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory_shadow"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_SHADOW_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(response.content):
                logger.warning("Forced tool_choice unsupported in shadow mode, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_SHADOW_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Shadow memory consolidation: LLM did not call save_memory_shadow "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_shadow_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Shadow memory consolidation: unexpected save_memory_shadow arguments")
                return self._fail_or_raw_archive(messages)
            
            if "history_entry" not in args or "canonical_memories" not in args:
                logger.warning("Shadow memory consolidation: payload missing required fields")
                return self._fail_or_raw_archive(messages)

            history_entry = _ensure_text(args["history_entry"]).strip()
            canonical_memories = args["canonical_memories"]
            
            if not history_entry or not isinstance(canonical_memories, list):
                logger.warning("Shadow memory consolidation: payload contains invalid required fields")
                return self._fail_or_raw_archive(messages)

            normalized_snapshot = self._normalize_snapshot_memories(canonical_memories)
            if canonical_memories and not normalized_snapshot:
                logger.warning("Shadow memory consolidation: no valid canonical memories after normalization")
                return self._fail_or_raw_archive(messages)

            payload = {
                "history_entry": history_entry,
                "canonical_memories": normalized_snapshot,
                "message_count": len(messages),
            }
            self._write_debug_payload(payload)
            self.db.initialize()

            now = datetime.now().isoformat(timespec="seconds")
            event_id = f"shadow_evt_{hashlib.sha1(f'{history_entry}|{now}'.encode('utf-8')).hexdigest()[:12]}"
            self.db.insert_raw_event(
                event_id=event_id,
                ts=now,
                session_key="shadow",
                history_text=history_entry,
                plain_text=self._format_messages(messages),
                extracted=payload,
                candidate_type="shadow_chunk",
            )
            self.db.replace_canonical_snapshot(normalized_snapshot, event_id=event_id)
            self.db.write_views()
            self._consecutive_failures = 0
            logger.info("Shadow memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Shadow memory consolidation failed")
            return self._fail_or_raw_archive(messages)
