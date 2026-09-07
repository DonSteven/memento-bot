"""Request budgeting at the actual model-call boundary."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from nanobot.agent.memory_db import MemoryContext
from nanobot.utils.helpers import estimate_prompt_tokens_chain


class ContextBudgetError(RuntimeError):
    """Required request content cannot fit without losing core memory."""


def request_tokens(provider, model, messages, tools) -> int:
    if not messages and not tools:
        return 0
    tokens, _ = estimate_prompt_tokens_chain(provider, model, messages, tools)
    if tokens <= 0:
        raise ContextBudgetError("Context budget cannot be determined; model call stopped.")
    return tokens


def fit_request(
    provider, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    context_window: int, output_tokens: int, memory_context: MemoryContext | None = None,
) -> MemoryContext | None:
    """Remove whole optional records/evidence, then reject an oversized request."""
    available = context_window - output_tokens
    while request_tokens(provider, model, messages, tools) > available:
        if memory_context and memory_context.retrieved_items:
            reduced = replace(memory_context, retrieved_items=memory_context.retrieved_items[:-1])
            old_block = memory_context.render_block()
            for message in messages:
                if message.get("role") == "system" and old_block in message.get("content", ""):
                    message["content"] = message["content"].replace(old_block, reduced.render_block(), 1)
                    memory_context = reduced
                    break
            else:
                raise ContextBudgetError("Prepared memory changed unexpectedly; model call stopped.")
            continue
        for message in messages:
            if message.get("role") != "tool" or message.get("name") != "kb_search":
                continue
            try:
                evidence = json.loads(message["content"])
            except (ValueError, TypeError):
                continue
            if not isinstance(evidence, dict) or not evidence.get("children"):
                continue
            evidence["children"].pop()
            parent_ids = {child["parent_id"] for child in evidence["children"]}
            evidence["parents"] = [p for p in evidence["parents"] if p["parent_id"] in parent_ids]
            evidence["budget_limited"] = True
            # The original assessment no longer describes the evidence actually sent.
            if evidence.get("status") in {"sufficient", "insufficient"}:
                evidence.update(status="insufficient", sufficient=False,
                                reason="Evidence reduced to fit context budget; coverage is not established.")
                evidence["missing_points"] = ["Complete evidence within the available context budget"]
            message["content"] = json.dumps(evidence, ensure_ascii=False)
            break
        else:
            required = request_tokens(provider, model, messages, tools) + output_tokens
            raise ContextBudgetError(
                f"Context limit exceeded: request plus output reserve needs {required} tokens, "
                f"limit is {context_window}. Core memory was preserved; model call stopped."
            )
    return memory_context
