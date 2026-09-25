"""Per-run hook for bounded, best-effort Agent observations."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.observability.events import DashboardEvents
from nanobot.observability.store import ObservabilityStore, utc_now

_SECRET_KEYS = {"apikey", "token", "password", "secret", "authorization",
                "accesskey", "accesstoken", "refreshtoken", "clientsecret", "cookie",
                "setcookie", "xapikey"}
_DATA_URL = re.compile(r"data:[^\s,;\"'<>]*(?:;[^\s,;\"'<>]+)*;base64,[A-Za-z0-9+/_=-]+",
                       re.IGNORECASE)
_ASSIGNMENT = re.compile(
    r"(?<![\w-])(?P<key>[A-Za-z][A-Za-z0-9_-]*)(?P<sep>\s*[:=]\s*)"
    r"(?P<value>\[REDACTED\]|\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;}\]]+)"
)
_AUTH_HEADER = re.compile(r"(?im)^(?P<prefix>[ \t]*Authorization[ \t]*:[ \t]*"
                          r"(?:Bearer|Basic)[ \t]+)[^\r\n]+")
_COOKIE_HEADER = re.compile(r"(?im)^(?P<prefix>[ \t]*(?:Cookie|Set-Cookie)[ \t]*:[ \t]*)"
                            r"[^\r\n]+")


def _sensitive_key(key: str) -> bool:
    return re.sub(r"[^a-z0-9]", "", key.lower()) in _SECRET_KEYS


def _redact_text(value: str) -> str:
    def replace_assignment(match: re.Match[str]) -> str:
        key, separator, content = match.group("key", "sep", "value")
        if not _sensitive_key(key) or content == "[REDACTED]":
            return match.group()
        # Preserve the scheme so the full Authorization header can be matched below.
        if key.lower() == "authorization" and content.lower() in {"bearer", "basic"}:
            return match.group()
        quote = content[0] if content.startswith(("'", '"')) else ""
        return f"{key}{separator}{quote}[REDACTED]{quote}"

    value = _DATA_URL.sub("[BINARY DATA]", value)
    value = _ASSIGNMENT.sub(replace_assignment, value)
    value = _AUTH_HEADER.sub(lambda match: match.group("prefix") + "[REDACTED]", value)
    return _COOKIE_HEADER.sub(lambda match: match.group("prefix") + "[REDACTED]", value)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if isinstance(key, str) and _sensitive_key(key)
                else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except ValueError:
                pass
            else:
                if isinstance(parsed, (dict, list)):
                    return json.dumps(redact(parsed), ensure_ascii=False, default=str)
        return _redact_text(value)
    return value


def preview(value: Any, max_bytes: int = 4096) -> dict[str, Any]:
    value = redact(value)
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    encoded = value.encode("utf-8", errors="replace")
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return {"text": clipped, "truncated": len(encoded) > max_bytes}


class ObservabilityHook(AgentHook):
    def __init__(self, store: ObservabilityStore, run_id: str,
                 events: DashboardEvents | None = None):
        self.store = store
        self.run_id = run_id
        self.events = events
        self._model_started: dict[int, tuple[str, float]] = {}
        self._tools_started: dict[int, tuple[str, float]] = {}
        self._model_recorded: set[int] = set()
        self._pending: set[asyncio.Task] = set()

    async def _write(self, method: str, *args: Any) -> None:
        task = asyncio.create_task(asyncio.to_thread(getattr(self.store, method), *args))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        try:
            await asyncio.shield(task)
            if self.events is not None:
                try:
                    self.events.publish("run.updated", run_id=self.run_id)
                except Exception:
                    logger.exception("Dashboard event broadcast failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Observability {} failed", method)

    async def drain(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._model_started[context.iteration] = (utc_now(), time.perf_counter())

    async def _model(self, context: AgentHookContext) -> None:
        if context.response is None or context.iteration in self._model_recorded:
            return
        self._model_recorded.add(context.iteration)
        started_at, start = self._model_started[context.iteration]
        content = preview(context.response.content or "", 1024)
        usage_reported = bool(context.response.usage)
        await self._write("add_event", self.run_id, "model", {
            "usage": context.usage, "usage_reported": usage_reported,
            "stop_reason": context.stop_reason or context.response.finish_reason,
            "tool_call_count": len(context.tool_calls), "content_preview": content,
        }, context.iteration, (time.perf_counter() - start) * 1000, started_at)
        await self._write("add_usage", self.run_id, context.usage.get("prompt_tokens", 0),
                          context.usage.get("completion_tokens", 0), usage_reported)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        await self._model(context)
        self._tools_started[context.iteration] = (utc_now(), time.perf_counter())

    async def after_iteration(self, context: AgentHookContext) -> None:
        await self._model(context)
        if context.tool_calls:
            calls = []
            for index, call in enumerate(context.tool_calls):
                event = context.tool_events[index] if index < len(context.tool_events) else {}
                result = context.tool_results[index] if index < len(context.tool_results) else None
                calls.append({
                    "call_id": call.id, "name": call.name,
                    "arguments": preview(call.arguments, 2048),
                    "status": event.get("status", "error"),
                    "result_preview": preview(result),
                })
            started_at, start = self._tools_started[context.iteration]
            await self._write("add_event", self.run_id, "tools", {"calls": calls},
                              context.iteration, (time.perf_counter() - start) * 1000,
                              started_at)
        if context.error and context.response is None:
            await self._write("add_event", self.run_id, "error",
                              {"error": preview(context.error, 1024)}, context.iteration)
