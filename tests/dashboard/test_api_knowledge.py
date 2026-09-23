from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import httpx
from aiohttp.test_utils import TestClient, TestServer

from nanobot.agent.knowledge import ChildEvidence, ParentEvidence
from nanobot.agent.knowledge_retrieval import KnowledgeResult
from nanobot.api.dashboard import create_dashboard_app
from nanobot.observability.store import ObservabilityStore
from nanobot.tests.knowledge.test_knowledge_online import setup_online


def app_for(retriever, tmp_path):
    agent = SimpleNamespace(knowledge_retriever=retriever)
    return create_dashboard_app(agent, object(), ObservabilityStore(tmp_path / "dashboard.db"))


def evidence():
    parent = ParentEvidence(10, 3, "Source", "https://example.com/source", "Available text",
                            False, ("fts", "vec"), 0.9, 1)
    child = ChildEvidence(20, 10, 3, "Source", "https://example.com/source", "Answer passage",
                          False, ("fts", "vec"), 0.9, 1, 1, 1, 0.88)
    return (parent,), (child,)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,sufficient", [
    ("sufficient", True), ("insufficient", False), ("retrieval_error", None),
    ("assessment_error", None), ("online_error", False),
])
async def test_knowledge_api_passes_through_result_and_limits(tmp_path, status, sufficient):
    parents, children = evidence()
    result = KnowledgeResult(
        query="question", status=status, sufficient=sufficient, reason="Final assessment",
        missing_points=("Missing detail",), parents=parents, children=children,
        online_attempted=True, fetched_urls=("https://example.com/attempt",),
        ingested_urls=(), online_errors=("Fetch failed",),
    )
    retrieve = AsyncMock(return_value=result)
    async with TestClient(TestServer(app_for(SimpleNamespace(retrieve=retrieve), tmp_path))) as client:
        retrieve.assert_not_awaited()
        response = await client.post("/api/dashboard/knowledge/search", json={
            "query": "question", "doc_limit": 2, "evidence_limit": 4,
        })
        assert response.status == 200
        assert await response.json() == result.to_dict()
        retrieve.assert_awaited_once_with("question", doc_limit=2, evidence_limit=4)


@pytest.mark.asyncio
async def test_knowledge_api_runs_real_online_path_with_fake_dependencies(tmp_path, setup_online):
    retriever, service, provider, search, fetch = setup_online
    async with TestClient(TestServer(app_for(retriever, tmp_path))) as client:
        search.assert_not_awaited()
        fetch.assert_not_awaited()
        response = await client.post("/api/dashboard/knowledge/search", json={
            "query": " Linux release ", "doc_limit": 3, "evidence_limit": 2,
        })
        assert response.status == 200
        result = await response.json()
        assert result["status"] == "sufficient"
        assert result["online_attempted"] is True
        assert result["fetched_urls"] == ["https://example.com/a"]
        assert result["ingested_urls"] == ["https://example.com/a"]
        assert result["children"][0]["parent_id"] == result["parents"][0]["parent_id"]
        assert len(service.db.list_pages()) == 1
        search.assert_awaited_once_with("Linux release")
        fetch.assert_awaited_once()
        assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream_error", [RuntimeError("upstream down"),
                                               httpx.ConnectError("network unavailable")])
async def test_knowledge_validation_unavailable_and_upstream_error(tmp_path, upstream_error):
    retrieve = AsyncMock(side_effect=upstream_error)
    async with TestClient(TestServer(app_for(SimpleNamespace(retrieve=retrieve), tmp_path))) as client:
        for body in (
            {}, {"query": " "}, {"query": "x" * 513}, {"query": "x", "doc_limit": 0},
            {"query": "x", "evidence_limit": True}, {"query": "x", "evidence_limit": 101},
            {"query": "x", "docLimit": 2},
        ):
            response = await client.post("/api/dashboard/knowledge/search", json=body)
            assert response.status == 400
        response = await client.post("/api/dashboard/knowledge/search", data="{bad",
                                     headers={"Content-Type": "application/json"})
        assert response.status == 400
        retrieve.assert_not_awaited()

        response = await client.post("/api/dashboard/knowledge/search", json={"query": "x"})
        assert response.status == 502
        assert (await response.json())["error"]["code"] == "knowledge_upstream_failed"
        retrieve.assert_awaited_once_with("x", doc_limit=None, evidence_limit=None)

    async with TestClient(TestServer(app_for(None, tmp_path))) as client:
        response = await client.post("/api/dashboard/knowledge/search", json={"query": "x"})
        assert response.status == 503
        assert (await response.json())["error"]["code"] == "knowledge_unavailable"
