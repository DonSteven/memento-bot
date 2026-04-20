"""Local external web knowledge search tool.
这是很薄的一层工具适配器"""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.knowledge import WebKnowledgeService
from nanobot.agent.tools.base import Tool


class KnowledgeSearchTool(Tool):
    """Search the local external web knowledge base before going online."""

    name = "kb_search"
    description = (
        "Search the local external web knowledge base built from prior successful web_fetch results. "
        "Returns candidate parent blocks plus child evidence chunks. "
        "Use this before web_search/web_fetch for non-current factual questions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Question or retrieval query to look up locally."},
            "docLimit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            "evidenceLimit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, service: WebKnowledgeService):
        self._service = service

    async def execute(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        result = await self._service.search(
            query,
            doc_limit=kwargs.get("docLimit"),
            evidence_limit=kwargs.get("evidenceLimit"),
        )
        return json.dumps(result, ensure_ascii=False)
