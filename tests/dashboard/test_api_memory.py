from dataclasses import asdict
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.agent.memory_db import MEMORY_CLASSES, MemoryRecord, MemorySnapshot
from nanobot.agent.memory_sync import MarkdownValidationError, MemoryConflictError
from nanobot.api.dashboard import create_dashboard_app
from nanobot.config.schema import MemoryConfig
from nanobot.observability.store import ObservabilityStore
from nanobot.tests.memory_test_utils import FixedEmbedding, memory_service


def make_service(tmp_path, *, top_k=2):
    embedder = FixedEmbedding()
    config = MemoryConfig.model_validate({
        "embedding": {"dimensions": 3, "model": "fixed-test"},
        "dynamic_top_k": top_k,
        "vector_similarity_threshold": 0.5,
    })
    service = memory_service(tmp_path / "workspace", AsyncMock(), embedder=embedder, config=config)
    records = [
        MemoryRecord.create("personal_profile", "name", "Ada"),
        MemoryRecord.create("preferences", "style", "Be concise"),
        MemoryRecord.create("constraints", "scope", "Local only"),
        MemoryRecord.create("projects", "active", "Nanobot retrieval engine"),
        MemoryRecord.create("daily_life", "drink", "Coffee every morning"),
        MemoryRecord.create("plans_commitments", "next", "Nanobot semantic search"),
    ]
    service.database.commit_snapshot(
        MemorySnapshot(0, tuple(records)), expected_revision=0, event_id="test-event",
        ts="2026-09-23T00:00:00", session_key="test", history_text="test",
        dynamic_embeddings={record.memory_id: embedder.vector(record.text)
                            for record in records if record.main_class in MEMORY_CLASSES[3:]},
    )
    return service, embedder


def app_for(service, tmp_path):
    agent = type("Agent", (), {"memory_service": service})()
    return create_dashboard_app(agent, object(), ObservabilityStore(tmp_path / "dashboard.db"))


@pytest.mark.asyncio
async def test_memory_snapshot_is_read_only_and_search_matches_service(tmp_path):
    service, embedder = make_service(tmp_path)
    revision = service.database.read_snapshot().revision
    async with TestClient(TestServer(app_for(service, tmp_path))) as client:
        response = await client.get("/api/dashboard/memory")
        assert response.status == 200
        data = await response.json()
        assert data["revision"] == revision
        assert data["counts"] == {main_class: 1 for main_class in MEMORY_CLASSES}
        assert data["records"] == [asdict(record) for record in service.database.read_snapshot().memories]
        assert embedder.document_calls == []
        assert embedder.query_calls == []
        assert service.database.read_snapshot().revision == revision

        response = await client.post("/api/dashboard/memory/search", json={"query": "Nanobot", "limit": 10})
        assert response.status == 200
        result = await response.json()
        assert result["core"] == [asdict(record) for record in service.database.read_core_memories()]
        expected = await service.search_dynamic("Nanobot", limit=10)
        assert result["dynamic"] == [
            {**asdict(hit), "sources": list(hit.sources)} for hit in expected
        ]
        assert len(result["dynamic"]) == service.config.dynamic_top_k
        assert embedder.query_calls


@pytest.mark.asyncio
async def test_memory_search_validation_and_error_mapping(tmp_path):
    service, embedder = make_service(tmp_path)
    async with TestClient(TestServer(app_for(service, tmp_path))) as client:
        for body in ({"query": " "}, {"query": "x" * 513}, {"query": "x", "limit": True},
                     {"query": "x", "limit": 0}, {"query": "x", "limit": 101}):
            response = await client.post("/api/dashboard/memory/search", json=body)
            assert response.status == 400
        assert embedder.query_calls == []

        original_sync = service.sync_markdown
        service.sync_markdown = AsyncMock(side_effect=MemoryConflictError("conflicting edit"))
        response = await client.post("/api/dashboard/memory/search", json={"query": "Nanobot"})
        assert response.status == 409
        assert (await response.json())["error"]["code"] == "memory_conflict"
        service.sync_markdown = AsyncMock(side_effect=MarkdownValidationError(2, "invalid record"))
        response = await client.post("/api/dashboard/memory/search", json={"query": "Nanobot"})
        assert response.status == 409
        assert (await response.json())["error"]["code"] == "invalid_markdown"
        service.sync_markdown = original_sync

        embedder.embed_query = AsyncMock(side_effect=RuntimeError("upstream unavailable"))
        response = await client.post("/api/dashboard/memory/search", json={"query": "Nanobot"})
        assert response.status == 502
        assert (await response.json())["error"]["code"] == "embedding_failed"

    async with TestClient(TestServer(create_dashboard_app(object(), object(),
                                                    ObservabilityStore(tmp_path / "other.db")))) as client:
        assert (await client.get("/api/dashboard/memory")).status == 503
        assert (await client.post("/api/dashboard/memory/search", json={"query": "x"})).status == 503
