from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nanobot.agent.knowledge_db import WebKnowledgeDatabase
from nanobot.agent.memory_db import MemoryDatabase, MemoryRecord, MemorySnapshot
from nanobot.agent.memory_service import MemoryService
from nanobot.agent.retrieval import build_fts_query, fts_index_text, fts_tokens
from nanobot.config.schema import MemoryConfig
from nanobot.tests.memory_test_utils import FixedEmbedding, memory_service


def _commit(db: MemoryDatabase, revision: int, records, vectors) -> int:
    return db.commit_snapshot(
        MemorySnapshot(revision, tuple(records)),
        expected_revision=revision,
        event_id=f"event-{revision}",
        ts=f"2026-09-05T00:00:0{revision}",
        session_key="test",
        history_text="test",
        dynamic_embeddings=vectors,
    )


def test_shared_fts_tokenizer_supports_chinese_mixed_punctuation_and_short_english() -> None:
    tokens = fts_tokens("AI，中文检索! go")
    assert {"ai", "中文", "文检", "检索", "中", "文", "检", "索", "go"} <= set(tokens)
    assert build_fts_query("纯中文")
    assert fts_index_text("中文") == "中文 中 文"


def test_chinese_fts_is_used_by_memory_and_knowledge_indexes(tmp_path) -> None:
    memory_db = MemoryDatabase(tmp_path / "memory-workspace", vec_backend="array")
    memory_db.initialize(3, embedding_provider="dashscope", embedding_model="fixed-test")
    item = MemoryRecord.create("projects", "active", "正在实现中文语义检索")
    _commit(memory_db, 0, [item], {item.memory_id: [1.0, 0.0, 0.0]})
    assert memory_db.search_dynamic_fts("中文检索", 5)[0]["memory_id"] == item.memory_id

    knowledge_db = WebKnowledgeDatabase(tmp_path / "knowledge-workspace", vec_backend="array")
    knowledge_db.initialize(3)
    knowledge_db.upsert_page_snapshot(
        source_url="https://example.com/cn",
        final_url="https://example.com/cn",
        title="中文说明",
        extractor="test",
        status=200,
        content_hash="one",
        raw_text="这个页面介绍中文检索。",
        is_partial=False,
        parents=["这个页面介绍中文检索。"],
        children=[{"parent_index": 0, "child_index": 0, "text": "这个页面介绍中文检索。"}],
        child_embeddings=[[1.0, 0.0, 0.0]],
        now="2026-09-05T00:00:00",
    )
    assert knowledge_db.search_child_fts("中文检索", 5)[0]["text"] == "这个页面介绍中文检索。"


@pytest.mark.asyncio
async def test_vector_only_hit_threshold_ranks_and_empty_query_short_circuit(tmp_path) -> None:
    embedder = FixedEmbedding()
    service = memory_service(tmp_path, AsyncMock(), embedder=embedder)
    related = MemoryRecord.create("projects", "active", "Build a retrieval engine")
    unrelated = MemoryRecord.create("daily_life", "drink", "Coffee every morning")
    _commit(
        service.database,
        0,
        [related, unrelated],
        {related.memory_id: [1.0, 0.0, 0.0], unrelated.memory_id: [0.0, 1.0, 0.0]},
    )

    hits = await service.search_dynamic("语义向量", limit=5)

    assert [hit.record.memory_id for hit in hits] == [related.memory_id]
    assert hits[0].sources == ("vector",)
    assert hits[0].vector_rank == 1
    assert hits[0].vector_similarity == pytest.approx(1.0)
    before = list(embedder.query_calls)
    assert await service.search_dynamic("") == ()
    assert embedder.query_calls == before


@pytest.mark.asyncio
async def test_update_delete_and_category_move_keep_both_indexes_in_sync(tmp_path) -> None:
    service = memory_service(tmp_path, AsyncMock())
    old = MemoryRecord.create("projects", "active", "旧版中文检索")
    _commit(service.database, 0, [old], {old.memory_id: [1.0, 0.0, 0.0]})
    core = MemoryRecord.create("preferences", "active", "旧版中文检索")
    _commit(service.database, 1, [core], {})

    assert service.database.search_dynamic_fts("旧版", 5) == []
    assert service.database.search_dynamic_vector([1.0, 0.0, 0.0], 5) == []
    with service.database.connect() as conn:
        assert conn.execute("select count(*) from memory_vector_map").fetchone()[0] == 0
        assert conn.execute("select count(*) from memory_vec").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_unchanged_snapshot_is_not_reembedded_and_budget_does_not_trim_core(tmp_path) -> None:
    embedder = FixedEmbedding()
    config = MemoryConfig.model_validate(
        {
            "embedding": {"dimensions": 3, "model": "fixed-test"},
            "dynamic_top_k": 3,
            "dynamic_token_budget": 1,
            "vector_similarity_threshold": 0.5,
        }
    )
    provider = AsyncMock()
    service = memory_service(tmp_path, provider, embedder=embedder, config=config)
    core = MemoryRecord.create("preferences", "style", "Always retain this complete core memory")
    dynamic = MemoryRecord.create("projects", "active", "Nanobot semantic retrieval")
    _commit(service.database, 0, [core, dynamic], {dynamic.memory_id: [1.0, 0.0, 0.0]})
    service.pipeline.extract_snapshot = AsyncMock(
        return_value=(service.database.read_snapshot(), "unchanged")
    )
    await service.sync_markdown()

    result = await service.consolidate(
        [{"role": "user", "content": "nothing changed"}], session_key="test"
    )
    context = await service.prepare_context("nanobot")

    assert result.database_committed
    assert embedder.document_calls == []
    assert context.core_items == (core,)
    assert context.retrieved_items == ()


def test_memory_index_metadata_rejects_same_dimension_different_model(tmp_path) -> None:
    db = MemoryDatabase(tmp_path, vec_backend="array")
    db.initialize(3, embedding_provider="dashscope", embedding_model="model-a")
    reopened = MemoryDatabase(tmp_path, vec_backend="array")
    with pytest.raises(RuntimeError, match="embedding_model.*Rebuild"):
        reopened.initialize(3, embedding_provider="dashscope", embedding_model="model-b")


def test_rebuild_indexes_replaces_derived_state_and_model_metadata(tmp_path) -> None:
    db = MemoryDatabase(tmp_path, vec_backend="array")
    db.initialize(3, embedding_provider="dashscope", embedding_model="model-a")
    item = MemoryRecord.create("projects", "active", "中文索引重建")
    _commit(db, 0, [item], {item.memory_id: [1.0, 0.0, 0.0]})

    db.rebuild_indexes(
        dynamic_embeddings={item.memory_id: [0.0, 1.0, 0.0]},
        vector_dim=3,
        embedding_provider="dashscope",
        embedding_model="model-b",
    )
    reopened = MemoryDatabase(tmp_path, vec_backend="array")
    reopened.initialize(3, embedding_provider="dashscope", embedding_model="model-b")

    assert reopened.search_dynamic_fts("中文索引", 5)[0]["memory_id"] == item.memory_id
    assert reopened.search_dynamic_vector([0.0, 1.0, 0.0], 1)[0]["memory_id"] == item.memory_id
    assert reopened.read_snapshot().revision == 1


def test_production_memory_requires_dashscope_credentials(tmp_path) -> None:
    with pytest.raises(ValueError, match="providers.dashscope.apiKey"):
        MemoryService(tmp_path, AsyncMock(), "test-model", api_key="")


def test_real_sqlite_vec_insert_search_update_and_delete(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.initialize(3, embedding_provider="dashscope", embedding_model="fixed-test")
    first = MemoryRecord.create("projects", "active", "First vector")
    _commit(db, 0, [first], {first.memory_id: [1.0, 0.0, 0.0]})
    assert db.search_dynamic_vector([1.0, 0.0, 0.0], 1)[0]["memory_id"] == first.memory_id

    second = MemoryRecord.create("projects", "active", "Second vector")
    _commit(db, 1, [second], {second.memory_id: [0.0, 1.0, 0.0]})
    assert db.search_dynamic_vector([0.0, 1.0, 0.0], 1)[0]["memory_id"] == second.memory_id
    _commit(db, 2, [], {})
    assert db.search_dynamic_vector([0.0, 1.0, 0.0], 5) == []
