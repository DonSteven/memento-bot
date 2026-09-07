"""External knowledge retrieval tool.
这是很薄的一层工具适配器"""

from __future__ import annotations

import json
from typing import Any

from nanobot.agent.knowledge_retrieval import KnowledgeRetriever
from nanobot.agent.tools.base import Tool


class KnowledgeSearchTool(Tool):
    """Retrieve and assess local and online external evidence."""

    name = "kb_search"
    description = (
        "Retrieve external knowledge with traceable parent/child evidence and coverage assessment. "
        "When local evidence is insufficient, automatically search once, fetch up to three distinct "
        "URLs, persist their text, and reassess. Reports remaining gaps and execution errors."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Question or retrieval query.",
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
