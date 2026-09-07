from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.knowledge import (
    ChildEvidence,
    LocalKnowledgeResult,
    ParentEvidence,
    WebKnowledgeService,
)
from nanobot.agent.knowledge_db import WebKnowledgeDatabase
from nanobot.agent.knowledge_retrieval import KnowledgeRetriever
from nanobot.agent.tools.knowledge import KnowledgeSearchTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.config.schema import KnowledgeConfig
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _AssessmentProvider(LLMProvider):
    def __init__(self, arguments: dict[str, Any] | None = None, *, error: bool = False) -> None:
        super().__init__()
        self.arguments = arguments
        self.error = error
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        if self.error:
            return LLMResponse(content="upstream unavailable", finish_reason="error")
        if self.arguments is None:
            return LLMResponse(content="not structured")
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="assessment-1",
                    name="report_evidence_assessment",
                    arguments=self.arguments,
                )
            ],
        )

    def get_default_model(self) -> str:
        return "test-model"


def _parent(*, partial: bool = False) -> ParentEvidence:
    return ParentEvidence(
        parent_id=10,
        page_id=3,
        title="Linux release",
        url="https://example.com/linux",
        text="Linux 6.9 was released in May 2024.",
        partial=partial,
        sources=("fts", "vec"),
        rerank_score=0.95,
        rerank_rank=1,
    )


def _child(*, score: float = 0.95, partial: bool = False) -> ChildEvidence:
    return ChildEvidence(
        child_id=20,
        parent_id=10,
        page_id=3,
        title="Linux release",
        url="https://example.com/linux",
        text="Linux 6.9 was released in May 2024.",
        partial=partial,
        sources=("fts", "vec"),
        rerank_score=score,
        rerank_rank=1,
        fts_rank=1,
        vector_rank=1,
        vector_similarity=0.88,
    )


def _retriever(local: LocalKnowledgeResult, provider: _AssessmentProvider, **config: Any):
    service = AsyncMock()
    service.search_local.return_value = local
    resolved = KnowledgeConfig(enabled=True, **config)
    return KnowledgeRetriever(
        search=SimpleNamespace(search=AsyncMock(return_value=[])), fetch=WebFetchTool(),
        service=service,
        provider=provider,
        model="test-model",
        config=resolved,
    ), service


@pytest.mark.asyncio
async def test_empty_library_is_insufficient_without_model_assessment() -> None:
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "yes"})
    retriever, _ = _retriever(LocalKnowledgeResult("question", (), ()), provider)

    result = await retriever.retrieve("question")

    assert result.status == "insufficient"
    assert result.sufficient is False
    assert provider.calls == []


@pytest.mark.asyncio
async def test_dual_source_top_one_below_threshold_is_still_insufficient() -> None:
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "yes"})
    retriever, _ = _retriever(
        LocalKnowledgeResult("question", (_parent(),), (_child(score=0.19),)), provider
    )

    result = await retriever.retrieve("question")

    assert result.status == "insufficient"
    assert result.children == ()
    assert provider.calls == []


@pytest.mark.asyncio
async def test_multiple_unrelated_candidates_do_not_bypass_relevance_filter() -> None:
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "yes"})
    first = _child(score=0.0)
    second = replace(first, child_id=21, rerank_rank=2, fts_rank=2, vector_rank=2)
    retriever, _ = _retriever(
        LocalKnowledgeResult("unrelated question", (_parent(),), (first, second)), provider
    )

    result = await retriever.retrieve("unrelated question")

    assert result.status == "insufficient"
    assert result.children == ()
    assert provider.calls == []


@pytest.mark.asyncio
async def test_evidence_that_does_not_fit_budget_is_not_sent_to_assessor() -> None:
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "yes"})
    retriever, _ = _retriever(
        LocalKnowledgeResult("question", (_parent(),), (_child(),)),
        provider,
        evidence_token_budget=1,
    )

    result = await retriever.retrieve("question")

    assert result.status == "insufficient"
    assert result.parents == ()
    assert result.children == ()
    assert provider.calls == []


@pytest.mark.asyncio
async def test_single_relevant_evidence_can_be_assessed_sufficient_and_is_traceable() -> None:
    provider = _AssessmentProvider(
        {"sufficient": True, "missing_points": [], "reason": "The release date is stated."}
    )
    retriever, _ = _retriever(
        LocalKnowledgeResult("When was Linux 6.9 released?", (_parent(),), (_child(),)), provider
    )

    result = await retriever.retrieve("When was Linux 6.9 released?")
    payload = result.to_dict()

    assert result.status == "sufficient"
    assert result.sufficient is True
    assert payload["children"][0]["parent_id"] == payload["parents"][0]["parent_id"]
    assert payload["children"][0]["page_id"] == payload["parents"][0]["page_id"]
    assert payload["children"][0]["url"] == "https://example.com/linux"
    assert payload["children"][0]["rerank_score"] == 0.95
    assert payload["children"][0]["sources"] == ["fts", "vec"]
    assert "untrusted data" in provider.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_partial_coverage_is_reported_with_missing_points_and_partial_source() -> None:
    provider = _AssessmentProvider(
        {
            "sufficient": False,
            "missing_points": ["The release location"],
            "reason": "Only the date is covered.",
        }
    )
    retriever, _ = _retriever(
        LocalKnowledgeResult("When and where?", (_parent(partial=True),), (_child(partial=True),)),
        provider,
    )

    result = await retriever.retrieve("When and where?")

    assert result.status == "insufficient"
    assert result.missing_points == ("The release location",)
    assert result.parents[0].partial is True
    assert result.children[0].partial is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [
        _AssessmentProvider(None),
        _AssessmentProvider({"sufficient": "yes", "missing_points": [], "reason": "bad"}),
        _AssessmentProvider(error=True),
    ],
)
async def test_invalid_or_failed_assessment_is_an_error_not_insufficient(provider) -> None:
    retriever, _ = _retriever(LocalKnowledgeResult("question", (_parent(),), (_child(),)), provider)

    result = await retriever.retrieve("question")

    assert result.status == "assessment_error"
    assert result.sufficient is None


@pytest.mark.asyncio
async def test_assessment_timeout_is_an_explicit_error() -> None:
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "yes"})

    async def _slow_chat(*args, **kwargs):
        await asyncio.sleep(0.05)
        return LLMResponse(content=None)

    provider.chat_with_retry = _slow_chat  # type: ignore[method-assign]
    retriever, _ = _retriever(
        LocalKnowledgeResult("question", (_parent(),), (_child(),)),
        provider,
        assessment_timeout_seconds=0.001,
    )

    result = await retriever.retrieve("question")

    assert result.status == "assessment_error"
    assert result.sufficient is None
    assert result.reason == "Evidence assessment timed out."


@pytest.mark.asyncio
async def test_local_retrieval_failure_has_distinct_error_status() -> None:
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "yes"})
    service = AsyncMock()
    service.search_local.side_effect = RuntimeError("reranker unavailable")
    retriever = KnowledgeRetriever(
        search=SimpleNamespace(search=AsyncMock(return_value=[])), fetch=WebFetchTool(),
        service=service,
        provider=provider,
        model="test-model",
        config=KnowledgeConfig(enabled=True),
    )

    result = await retriever.retrieve("question")

    assert result.status == "retrieval_error"
    assert result.sufficient is None
    assert provider.calls == []


def _indexed_service(tmp_path, pages, **settings):
    config = KnowledgeConfig(enabled=True, **settings)
    scores = {text: score for _, children in pages for text, score in children}

    async def score_pairs(pairs):
        return [scores[text] for _, text in pairs]

    service = WebKnowledgeService(
        workspace=tmp_path,
        config=config,
        db=WebKnowledgeDatabase(tmp_path, vec_backend="array"),
        embedder=SimpleNamespace(dimension=2, embed_query=AsyncMock(return_value=[1.0, 0.0])),
        reranker=SimpleNamespace(score_pairs=score_pairs),
    )
    for index, (parent_text, children) in enumerate(pages):
        service.db.upsert_page_snapshot(
            source_url=f"https://example.com/{index}",
            final_url=f"https://example.com/{index}",
            title=f"Page {index}",
            extractor="test",
            status=200,
            content_hash=str(index),
            raw_text=parent_text,
            is_partial=False,
            parents=[parent_text],
            children=[
                {"parent_index": 0, "child_index": child_index, "text": text}
                for child_index, (text, _) in enumerate(children)
            ],
            child_embeddings=[[1.0, 0.0] for _ in children],
            now="2026-09-07T00:00:00",
        )
    return service


@pytest.mark.asyncio
async def test_low_score_children_do_not_exclude_other_relevant_parents(tmp_path) -> None:
    service = _indexed_service(
        tmp_path,
        [("Answer A and unrelated text", [("Answer A", 0.9), ("Unrelated", 0.1)]),
         ("Answer B", [("Answer B", 0.8)])],
    )
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "A and B"})
    retriever = KnowledgeRetriever(
        search=SimpleNamespace(search=AsyncMock(return_value=[])), fetch=WebFetchTool(),
        service=service, provider=provider, model="test-model", config=service.config
    )

    result = await retriever.retrieve("A and B?", evidence_limit=2)

    assert [child.text for child in result.children] == ["Answer A", "Answer B"]
    assert len(result.parents) == 2
    assessed = json.loads(provider.calls[0][1]["content"].split("Available evidence:\n", 1)[1])
    assert assessed == {"parents": result.to_dict()["parents"], "children": result.to_dict()["children"]}


@pytest.mark.asyncio
async def test_over_budget_parent_does_not_consume_document_or_evidence_slot(tmp_path) -> None:
    service = _indexed_service(
        tmp_path,
        [("Long context " * 10000, [("Answer in long context", 0.9)]),
         ("Short answer", [("Short answer", 0.8)])],
        evidence_token_budget=1000,
    )
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "covered"})
    retriever = KnowledgeRetriever(
        search=SimpleNamespace(search=AsyncMock(return_value=[])), fetch=WebFetchTool(),
        service=service, provider=provider, model="test-model", config=service.config
    )

    result = await retriever.retrieve("question", doc_limit=1, evidence_limit=1)

    assert [child.text for child in result.children] == ["Short answer"]
    assert [parent.text for parent in result.parents] == ["Short answer"]


@pytest.mark.asyncio
async def test_over_budget_child_does_not_consume_per_parent_slot(tmp_path) -> None:
    long_text = "Detailed answer " * 2000
    service = _indexed_service(
        tmp_path,
        [(long_text + " Short answer", [(long_text, 0.9), ("Short answer", 0.8)])],
        evidence_token_budget=6000,
        max_children_per_parent=1,
    )
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "covered"})
    retriever = KnowledgeRetriever(
        search=SimpleNamespace(search=AsyncMock(return_value=[])), fetch=WebFetchTool(),
        service=service, provider=provider, model="test-model", config=service.config
    )

    result = await retriever.retrieve("question", evidence_limit=1)

    assert [child.text for child in result.children] == ["Short answer"]


@pytest.mark.asyncio
async def test_output_limits_apply_to_selected_evidence(tmp_path) -> None:
    service = _indexed_service(
        tmp_path,
        [("A", [("A1", 0.9), ("A2", 0.8), ("A3", 0.7)]),
         ("B", [("B1", 0.6), ("B2", 0.5)]),
         ("C", [("C1", 0.4)])],
        doc_limit=2, evidence_limit=3, max_children_per_parent=2,
    )
    provider = _AssessmentProvider({"sufficient": True, "missing_points": [], "reason": "covered"})
    retriever = KnowledgeRetriever(
        search=SimpleNamespace(search=AsyncMock(return_value=[])), fetch=WebFetchTool(),
        service=service, provider=provider, model="test-model", config=service.config
    )

    result = await retriever.retrieve("question")

    assert [child.text for child in result.children] == ["A1", "A2", "B1"]
    result = await retriever.retrieve("question", doc_limit=1, evidence_limit=5)
    assert [child.text for child in result.children] == ["A1", "A2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("sufficient", [True])
async def test_kb_search_sufficient_local_pipeline_never_searches_or_fetches(
    tmp_path, monkeypatch, sufficient
) -> None:
    service = _indexed_service(tmp_path, [("Answer", [("Answer", 0.9)])])
    provider = _AssessmentProvider({
        "sufficient": sufficient,
        "missing_points": [] if sufficient else ["More detail"],
        "reason": "covered" if sufficient else "Incomplete coverage",
    })
    retriever = KnowledgeRetriever(
        search=SimpleNamespace(search=AsyncMock(return_value=[])), fetch=WebFetchTool(),
        service=service, provider=provider, model="test-model", config=service.config
    )
    search = AsyncMock(side_effect=AssertionError("Unexpected web search"))
    fetch = AsyncMock(side_effect=AssertionError("Unexpected web fetch"))
    monkeypatch.setattr(WebSearchTool, "execute", search)
    monkeypatch.setattr(WebFetchTool, "execute", fetch)
    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool(retriever))
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())

    result = json.loads(await registry.execute("kb_search", {"query": "question"}))

    assert result["status"] == ("sufficient" if sufficient else "insufficient")
    assert len(result["children"]) == 1
    search.assert_not_called()
    fetch.assert_not_called()
