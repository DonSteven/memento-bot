"""Side-effect-free structured memory extraction."""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.memory_db import MemoryRecord, MemorySnapshot, _canonical_memory_id
from nanobot.providers.base import LLMProvider

_SAVE_MEMORY_STRUCTURED_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory_structured",
            "description": "Return the complete updated memory snapshot and a history summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A grep-friendly summary beginning with [YYYY-MM-DD HH:MM].",
                    },
                    "canonical_memories": {
                        "type": "array",
                        "description": "The complete updated memory. Preserve all still-valid facts, including multiple facts in the same subclass.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "main_class": {
                                    "type": "string",
                                    "enum": [
                                        "personal_profile", "preferences", "constraints",
                                        "projects", "daily_life", "plans_commitments",
                                    ],
                                },
                                "sub_class": {
                                    "type": "string",
                                    "description": "A non-empty single-line label without the delimiter ': '.",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "One non-empty single-line fact, without line breaks.",
                                },
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
_TOOL_CHOICE_ERROR_MARKERS = ("tool_choice", "toolchoice", "does not support", 'should be ["none", "auto"]')


class MemoryExtractionError(RuntimeError):
    """The model did not return a valid complete memory snapshot."""


def _ensure_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_args(args: Any) -> dict[str, Any] | None:
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None


def format_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        content = message.get("content")
        if not content:
            continue
        raw_tools = message.get("tools_used") or []
        tools = f" [tools: {', '.join(str(tool) for tool in raw_tools)}]" if raw_tools else ""
        timestamp = str(message.get("timestamp") or "?")[:16]
        lines.append(f"[{timestamp}] {str(message.get('role') or '?').upper()}{tools}: {_ensure_text(content)}")
    return "\n".join(lines)


class StructuredMemoryPipeline:
    """Ask the model for a complete snapshot without touching storage or files."""

    def __init__(self, provider: LLMProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    async def extract_snapshot(
        self,
        messages: list[dict[str, Any]],
        current_snapshot: MemorySnapshot,
    ) -> tuple[MemorySnapshot, str]:
        if not messages:
            raise MemoryExtractionError("Cannot extract memory from an empty message list")
        current = "\n".join(
            f"- [{item.main_class}/{item.sub_class}] {item.text}"
            for item in current_snapshot.memories
        ) or "(empty)"
        prompt = (
            "Process the conversation and call save_memory_structured.\n\n"
            f"## Current Long-term Memory\n{current}\n\n"
            f"## Conversation to Process\n{format_messages(messages)}"
        )
        chat_messages = [
            {"role": "system", "content": "You are a memory consolidation agent. Return the complete structured memory snapshot through the required tool."},
            {"role": "user", "content": prompt},
        ]
        forced = {"type": "function", "function": {"name": "save_memory_structured"}}
        response = await self.provider.chat_with_retry(
            messages=chat_messages,
            tools=_SAVE_MEMORY_STRUCTURED_TOOL,
            model=self.model,
            tool_choice=forced,
        )
        if response.finish_reason == "error" and any(
            marker in (response.content or "").lower() for marker in _TOOL_CHOICE_ERROR_MARKERS
        ):
            response = await self.provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_STRUCTURED_TOOL,
                model=self.model,
                tool_choice="auto",
            )
        if not response.has_tool_calls:
            raise MemoryExtractionError("Model did not call save_memory_structured")
        try:
            args = _normalize_args(response.tool_calls[0].arguments)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MemoryExtractionError("Invalid save_memory_structured arguments") from exc
        if args is None or "history_entry" not in args or "canonical_memories" not in args:
            raise MemoryExtractionError("Structured memory response is missing required fields")
        if args["history_entry"] is None:
            raise MemoryExtractionError("Structured memory response contains a null history entry")
        history_entry = _ensure_text(args["history_entry"]).strip()
        raw_memories = args["canonical_memories"]
        if not history_entry or not isinstance(raw_memories, list):
            raise MemoryExtractionError("Structured memory response contains invalid required fields")

        records: dict[str, MemoryRecord] = {}
        try:
            for raw in raw_memories:
                if not isinstance(raw, dict):
                    raise ValueError("memory item must be an object")
                if raw.get("sub_class") is None or raw.get("text") is None:
                    raise ValueError("memory fields must not be null")
                main_class = str(raw.get("main_class") or "").strip()
                sub_class = _ensure_text(raw.get("sub_class", "")).strip()
                text = _ensure_text(raw.get("text", "")).strip()
                if not main_class or not sub_class or not text:
                    raise ValueError("memory fields must be non-empty")
                memory_id = _canonical_memory_id(main_class, sub_class, text)
                records[memory_id] = MemoryRecord(memory_id, main_class, sub_class, text)
        except (TypeError, ValueError) as exc:
            raise MemoryExtractionError(f"Invalid canonical memory: {exc}") from exc

        return MemorySnapshot(current_snapshot.revision, tuple(records.values())), history_entry
