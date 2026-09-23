"""Per-run hook for bounded, best-effort Agent observations."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.observability.store import ObservabilityStore, utc_now

_SECRET_KEYS = {"api_key", "apikey", "token", "password", "secret", "authorization",
                "access_key", "client_secret", "cookie"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower().replace("-", "_") in _SECRET_KEYS
                else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str) and (value.startswith("data:") and ";base64," in value):
        return "[BINARY DATA]"
    return value


def preview(value: Any, max_bytes: int = 4096) -> dict[str, Any]:
    if not isinstance(value, str):
        value = json.dumps(redact(value), ensure_ascii=False, default=str)
    encoded = value.encode("utf-8", errors="replace")
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return {"text": clipped, "truncated": len(encoded) > max_bytes}


class ObservabilityHook(AgentHook):
    def __init__(self, store: ObservabilityStore, run_id: str):
        self.store = store
        self.run_id = run_id
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
