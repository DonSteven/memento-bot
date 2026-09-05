# Knowledge backend test content:
# - verifies DashScope embedding batching, text types, vector ordering, and normalization
# - verifies qwen3-rerank endpoint selection and score ordering
# How to test:
# - uv run --extra dev pytest -q nanobot/tests/knowledge/test_dashscope_knowledge_backends.py

from __future__ import annotations

from typing import Any

import httpx
import pytest

from nanobot.agent.knowledge import _DashScopeEmbeddingBackend, _DashScopeRerankerBackend


class _FakeClient:
    def __init__(self, calls: list[dict[str, Any]], timeout: float):
        assert timeout == 30.0
        self.calls = calls

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any]
    ) -> httpx.Response:
        self.calls.append({"url": url, "headers": headers, "json": json})
        request = httpx.Request("POST", url)
        if url.endswith("/text-embedding"):
            texts = json["input"]["texts"]
            items = [
                {"text_index": index, "embedding": [3.0, 4.0]}
                for index in reversed(range(len(texts)))
            ]
            return httpx.Response(200, request=request, json={"output": {"embeddings": items}})
        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        )


@pytest.mark.asyncio
async def test_dashscope_embedding_batches_and_distinguishes_documents_from_query(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "nanobot.agent.knowledge.httpx.AsyncClient",
        lambda *, timeout: _FakeClient(calls, timeout),
    )
    backend = _DashScopeEmbeddingBackend(
        api_key="sk-test",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="text-embedding-v4",
        dimension=2,
    )

    vectors = await backend.embed_documents([f"document-{index}" for index in range(11)])
    query_vector = await backend.embed_query("query")

    assert backend.endpoint == (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "text-embedding/text-embedding"
    )
    assert len(vectors) == 11
    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert query_vector == pytest.approx([0.6, 0.8])
    assert [len(call["json"]["input"]["texts"]) for call in calls] == [10, 1, 1]
    assert [call["json"]["parameters"]["text_type"] for call in calls] == [
        "document",
        "document",
        "query",
    ]
    assert all(call["headers"]["Authorization"] == "Bearer sk-test" for call in calls)


@pytest.mark.asyncio
async def test_dashscope_reranker_uses_dedicated_endpoint_and_restores_input_order(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "nanobot.agent.knowledge.httpx.AsyncClient",
        lambda *, timeout: _FakeClient(calls, timeout),
    )
    backend = _DashScopeRerankerBackend(
        api_key="sk-test",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3-rerank",
        instruct="Retrieve relevant passages.",
    )

    scores = await backend.score_pairs([("query", "first"), ("query", "second")])

    assert backend.endpoint == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    assert scores == [0.1, 0.9]
    assert calls[0]["json"] == {
        "model": "qwen3-rerank",
        "query": "query",
        "documents": ["first", "second"],
        "top_n": 2,
        "instruct": "Retrieve relevant passages.",
    }
