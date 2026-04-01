"""Shadow memory pipeline for Phase 2 structured sidecar writes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.memory_db import MemoryDatabase
from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


_SAVE_MEMORY_SHADOW_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory_shadow",
            "description": "Extract a structured shadow-memory view of the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A grep-friendly history summary starting with [YYYY-MM-DD HH:MM].",
                    },
                    "canonical_memories": {
                        "type": "array",
                        "description": "Structured canonical memories extracted from the conversation.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "main_class": {
                                    "type": "string",
                                    "enum": [
                                        "user_info",
                                        "preferences",
                                        "project_context",
                                        "important_notes",
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


def _normalize_shadow_args(args: Any) -> dict[str, Any] | None:
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


class ShadowMemoryPipeline:
    """Best-effort sidecar pipeline that writes structured shadow memory artifacts."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.shadow_dir = ensure_dir(workspace / "memory" / "shadow")
        self.db = MemoryDatabase(
            workspace,
            storage_dir=self.shadow_dir,
            view_dir=self.shadow_dir,
        )
        self.debug_file = self.shadow_dir / "last_payload.json"

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

    @staticmethod
    def _memory_id(main_class: str, sub_class: str, text: str) -> str:
        slot = (sub_class or "").strip().lower()
        if slot:
            safe_slot = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in slot)
            return f"{main_class}:{safe_slot}"
        digest = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()[:12]
        return f"{main_class}:{digest}"

    def _write_debug_payload(self, payload: dict[str, Any]) -> None:
        self.debug_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
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
                Process this conversation and call the save_memory_shadow tool.

                ## Current Shadow Memory View
                {current_shadow_memory}

                ## Conversation to Process
                {self._format_messages(messages)}
                """

        chat_messages = [
            {
                "role": "system",
                "content": "You are a shadow memory extraction agent. Call the save_memory_shadow tool with structured memory candidates.",
            },
            {"role": "user", "content": prompt},
        ]

        forced = {"type": "function", "function": {"name": "save_memory_shadow"}}
        response = await provider.chat_with_retry(
            messages=chat_messages,
            tools=_SAVE_MEMORY_SHADOW_TOOL,
            model=model,
            tool_choice=forced,
        )

        if not response.has_tool_calls:
            logger.warning(
                "Shadow memory extraction skipped: no save_memory_shadow tool call returned"
            )
            return False

        args = _normalize_shadow_args(response.tool_calls[0].arguments)
        if args is None:
            logger.warning("Shadow memory extraction skipped: invalid tool arguments")
            return False

        history_entry = str(args.get("history_entry") or "").strip()
        canonical_memories = args.get("canonical_memories")
        if not history_entry or not isinstance(canonical_memories, list):
            logger.warning("Shadow memory extraction skipped: missing required payload fields")
            return False

        payload = {
            "history_entry": history_entry,
            "canonical_memories": canonical_memories,
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

        for item in canonical_memories:
            if not isinstance(item, dict):
                continue
            main_class = str(item.get("main_class") or "").strip()
            text = str(item.get("text") or "").strip()
            if not main_class or not text:
                continue
            sub_class = str(item.get("sub_class") or "").strip()
            confidence = float(item.get("confidence", 0.0) or 0.0)
            memory_id = self._memory_id(main_class, sub_class, text)
            self.db.upsert_canonical_memory(
                memory_id=memory_id,
                main_class=main_class,
                sub_class=sub_class,
                text=text,
                confidence=confidence,
            )
            self.db.add_evidence_link(memory_id=memory_id, event_id=event_id, weight=1.0)

        self.db.write_views()
        return True
