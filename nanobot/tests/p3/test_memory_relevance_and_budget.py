from unittest.mock import AsyncMock

import pytest
import tiktoken

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory_db import MemoryContext, MemoryRecord, MemorySnapshot
from nanobot.agent.retrieval import lexical_query_terms, passes_lexical_gate
from nanobot.config.schema import MemoryConfig
from nanobot.tests.memory_test_utils import memory_service


def _seed(service, records, vectors):
    service.database.commit_snapshot(
        MemorySnapshot(0, tuple(records)),
        expected_revision=0,
        event_id="seed",
        ts="2026-09-05",
        session_key="test",
        history_text="seed",
        dynamic_embeddings=vectors,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected_count"),
    [
        ({}, 8),
        ({"retrieval_budget": None}, 8),
        ({"retrieval_budget": 3}, 3),
        ({"retrieval_budget": 12}, 8),
        ({"retrieval_budget": 0}, 0),
        ({"retrieval_budget": -1}, 0),
    ],
)
async def test_context_uses_configured_top_k_unless_explicitly_tightened(
    tmp_path,
    kwargs,
    expected_count,
):
    config = MemoryConfig.model_validate(
        {
            "embedding": {"dimensions": 3, "model": "fixed-test"},
            "dynamic_top_k": 8,
        }
    )
    service = memory_service(tmp_path, AsyncMock(), config=config)
    core = MemoryRecord.create("preferences", "style", "Always retain core memory")
    records = tuple(
        MemoryRecord.create("projects", "active", f"Nanobot task {index}") for index in range(10)
    )
    _seed(service, (core, *records), {item.memory_id: [1.0, 0.0, 0.0] for item in records})

    prepared = await service.prepare_context("Nanobot", **kwargs)

    assert len(prepared.retrieved_items) == expected_count
    assert prepared.core_items == (core,)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_core", [False, True])
@pytest.mark.parametrize(
    "text",
    [
        "正在开发中文语义检索系统。" * 20,
        "Building a semantic retrieval system. " * 20,
        "Nanobot 使用 SQLite 实现 semantic retrieval。" * 20,
        "مرحبا Ελληνικά café 🚀 👨‍👩‍👧‍👦 " * 20,
        "Literal <|endoftext|> and e\u0301 are ordinary memory text.",
    ],
)
async def test_budget_counts_the_actual_injected_block(tmp_path, text, with_core):
    encoding = tiktoken.get_encoding("cl100k_base")
    core = (
        (MemoryRecord.create("preferences", "style", "完整保留核心记忆" * 200),)
        if with_core
        else ()
    )
    record = MemoryRecord.create("projects", "active", text)
    expected = MemoryContext(core, (record,))
    core_block = MemoryContext(core_items=core).render_block()
    dynamic_block = expected.render_block()[len(core_block) :]
    exact_budget = len(encoding.encode(dynamic_block, disallowed_special=()))
    config = MemoryConfig.model_validate(
        {
            "embedding": {"dimensions": 3, "model": "fixed-test"},
            "dynamic_token_budget": exact_budget,
        }
    )
    service = memory_service(tmp_path, AsyncMock(), config=config)
    _seed(service, (*core, record), {record.memory_id: [1.0, 0.0, 0.0]})

    prepared = await service.prepare_context("向量检索")
    assert prepared == expected
    prompt = ContextBuilder(tmp_path).build_system_prompt(memory_context=prepared)
    assert prepared.render_block() in prompt
    assert (
        len(encoding.encode(prepared.render_block()[len(core_block) :], disallowed_special=()))
        == exact_budget
    )

    service.config.dynamic_token_budget = exact_budget - 1
    limited = await service.prepare_context("向量检索")
    assert limited.core_items == core
    assert limited.retrieved_items == ()


def test_budget_counts_combined_rows_and_skips_oversized_candidates(tmp_path):
    service = memory_service(tmp_path, AsyncMock())
    first = MemoryRecord.create("projects", "active", "第一条")
    large = MemoryRecord.create("projects", "active", "超长内容" * 100)
    last = MemoryRecord.create("daily_life", "note", "Last entry")
    block = MemoryContext(retrieved_items=(first, last)).render_block()
    encoding = tiktoken.get_encoding("cl100k_base")
    budget = len(encoding.encode(block))
    selected = service._apply_token_budget((first, large, last), budget)
    assert selected == (first, last)
    assert len(encoding.encode(MemoryContext(retrieved_items=selected).render_block())) <= budget
    assert service._apply_token_budget((first,), 0) == ()


@pytest.mark.parametrize(
    ("query", "document", "expected"),
    [
        ("向量检索", "向朋友借书", False),
        ("向量检索", "向量索引", False),
        ("向量检索", "正在开发向量检索", True),
        ("向", "向朋友借书", True),
        ("AI 中文检索", "AI 中文检索系统", True),
        ("go", "Use Go", True),
    ],
)
def test_lexical_gate_uses_meaningful_term_coverage(query, document, expected):
    assert passes_lexical_gate(query, document) is expected
    assert "向" not in lexical_query_terms("向量检索")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "vector", "accepted"),
    [
        ("向朋友借书", [0.0, 0.0, 1.0], False),
        ("向量索引", [0.0, 0.0, 1.0], False),
        ("正在开发向量检索", [0.0, 0.0, 1.0], True),
        ("向朋友借书", [1.0, 0.0, 0.0], True),
    ],
)
async def test_candidate_must_pass_lexical_or_vector_gate(tmp_path, text, vector, accepted):
    service = memory_service(tmp_path, AsyncMock())
    record = MemoryRecord.create("projects", "active", text)
    _seed(service, (record,), {record.memory_id: vector})
    hits = await service.search_dynamic("向量检索")
    assert bool(hits) is accepted
    prepared = await service.prepare_context("向量检索")
    assert bool(prepared.retrieved_items) is accepted
