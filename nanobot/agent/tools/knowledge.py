"""Local external web knowledge search tool.
这是很薄的一层工具适配器"""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.knowledge_retrieval import KnowledgeRetriever
from nanobot.agent.tools.base import Tool


class KnowledgeSearchTool(Tool):
    """Retrieve and assess evidence from the local external knowledge base."""

    name = "kb_search"
    description = (
        "Search the local external web knowledge base built from prior successful web_fetch results. "
        "Returns traceable parent/child evidence, relevance scores, and an answer-coverage assessment. "
        "An assessment_error means sufficiency could not be determined. "
        "Use this before web_search/web_fetch for non-current factual questions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Question or retrieval query to look up locally.",
            },
            "docLimit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            "evidenceLimit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: KnowledgeRetriever):
        self._retriever = retriever

    async def execute(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        result = await self._retriever.retrieve(
            query,
            doc_limit=kwargs.get("docLimit"),
            evidence_limit=kwargs.get("evidenceLimit"),
        )
        return json.dumps(result.to_dict(), ensure_ascii=False)
