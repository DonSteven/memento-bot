import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.knowledge_db import WebKnowledgeDatabase
from nanobot.agent.memory_db import MemoryDatabase, MemoryRecord, MemorySnapshot
from nanobot.agent.memory_sync import render_memory_markdown
from nanobot.config.schema import Config
from nanobot.tests.knowledge.test_knowledge_retrieval import _indexed_service
from nanobot.tests.memory_test_utils import FixedEmbedding, TestMemoryService
from scripts.upgrade_memory_knowledge import convert_workspace, inspect_workspace


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    service = TestMemoryService(root, None)
    records = (MemoryRecord.create("constraints", "rule", "Keep all core facts"),
               MemoryRecord.create("projects", "active", "Nanobot 语义检索"))
    service.database.commit_snapshot(MemorySnapshot(0, records), expected_revision=0, event_id="event",
        ts="2026-09-07", session_key="cli:test", history_text="Deleted old project; kept new project",
        plain_text="full original messages", dynamic_embeddings={records[1].memory_id: [1., 0., 0.]})
    service.database.memory_file.write_text(render_memory_markdown(records), encoding="utf-8")
    _indexed_service(root, [("source page", [("source child", 0.9)])])
    (root / "notes.txt").write_text("preserve unrelated user files")
    return root


def config():
    return Config.model_validate({"memory": {"embedding": {"dimensions": 3, "model": "fixed-test"}},
                                  "knowledge": {"embedding": {"dimensions": 2}}})


def bytes_before(source):
    return {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}


@pytest.mark.asyncio
@pytest.mark.parametrize("old_schema", [False, True])
async def test_offline_conversion_preserves_facts_events_pages_and_real_vector_indexes(source, tmp_path, old_schema):
    if old_schema:
        # v5 memory and v3 knowledge have the same authoritative row layouts as current versions.
        with sqlite3.connect(source / "memory/memory.db") as conn:
            conn.execute("update metadata set value='5' where key='schema_version'")
            conn.execute("drop table memory_sync_state")
        with sqlite3.connect(source / "knowledge/web_knowledge.db") as conn:
            conn.execute("update metadata set value='3' where key='schema_version'")
    before = bytes_before(source)
    initial = inspect_workspace(source)
    destination = tmp_path / "converted"
    report = await convert_workspace(source, destination, config=config(), memory_embedder=FixedEmbedding(),
        knowledge_embedder=SimpleNamespace(embed_documents=AsyncMock(return_value=[[1., 0.]])))
    assert report["ready"] and not report["pending_embeddings"]
    assert bytes_before(source) == before
    final = inspect_workspace(destination)
    assert final["memory"]["raw_events"] == initial["memory"]["raw_events"]
    assert final["memory"]["canonical_memories"] == initial["memory"]["canonical_memories"]
    for table in ("web_pages", "page_parents", "parent_children"):
        assert final["knowledge"][table] == initial["knowledge"][table]
    db = MemoryDatabase(destination)
    db.initialize(3, embedding_provider="dashscope", embedding_model="fixed-test")
    assert db.search_dynamic_fts("语义", 5)
    assert db.search_dynamic_vector([1., 0., 0.], 5)
    assert len(db.read_snapshot().memories) == 2  # History never resurrects deleted facts.
    kb = WebKnowledgeDatabase(destination)
    kb.initialize(2)
    assert kb.search_child_fts("source", 5)
    assert kb.search_child_vector([1., 0.], 5)
    assert (destination / "notes.txt").read_text() == "preserve unrelated user files"


@pytest.mark.asyncio
async def test_conflict_requires_explicit_source_and_markdown_deletion_wins(source, tmp_path):
    core = MemoryRecord.create("constraints", "rule", "Keep all core facts")
    (source / "memory/MEMORY.md").write_text(render_memory_markdown((core,)))
    before = bytes_before(source)
    assert inspect_workspace(source)["report"]["source_choice_required"]
    with pytest.raises(ValueError, match="choose"):
        await convert_workspace(source, tmp_path / "blocked", config=config())
    assert not (tmp_path / "blocked").exists()
    await convert_workspace(source, tmp_path / "chosen", config=config(), memory_source="markdown")
    assert len(inspect_workspace(tmp_path / "chosen")["records"]) == 1
    assert bytes_before(source) == before


@pytest.mark.asyncio
async def test_no_embeddings_marks_copy_unusable_until_explicit_rebuild(source, tmp_path):
    report = await convert_workspace(source, tmp_path / "pending", config=config())
    assert not report["ready"] and set(report["pending_embeddings"]) == {"memory", "knowledge"}
    with pytest.raises(RuntimeError, match="mismatch"):
        MemoryDatabase(tmp_path / "pending").initialize(3, embedding_provider="dashscope", embedding_model="fixed-test")
    # Rebuild into a second fresh copy, preserving the earlier originals too.
    final = await convert_workspace(tmp_path / "pending", tmp_path / "ready", config=config(),
        memory_embedder=FixedEmbedding(), knowledge_embedder=SimpleNamespace(embed_documents=AsyncMock(return_value=[[1., 0.]])))
    assert final["ready"]


@pytest.mark.asyncio
async def test_embedding_failure_leaves_source_and_destination_untouched(source, tmp_path):
    before = bytes_before(source)
    with pytest.raises(RuntimeError, match="API failed"):
        await convert_workspace(source, tmp_path / "failed", config=config(),
            memory_embedder=SimpleNamespace(embed_documents=AsyncMock(side_effect=RuntimeError("API failed"))))
    assert bytes_before(source) == before
    assert not (tmp_path / "failed").exists()


@pytest.mark.asyncio
async def test_invalid_markdown_reports_line_and_refuses_conversion(source, tmp_path):
    path = source / "memory/MEMORY.md"
    path.write_text(path.read_text() + "\n- invalid")
    before = bytes_before(source)
    assert "line" in inspect_workspace(source)["report"]["problems"][0]
    with pytest.raises(ValueError, match="line"):
        await convert_workspace(source, tmp_path / "invalid", config=config())
    assert bytes_before(source) == before


@pytest.mark.asyncio
async def test_pending_publish_uses_committed_database_not_stale_markdown(source, tmp_path):
    db = MemoryDatabase(source, vec_backend="array")
    db.initialize(3, embedding_provider="dashscope", embedding_model="fixed-test")
    old = db.memory_file.read_text()
    new = MemoryRecord.create("constraints", "rule", "New committed fact")
    db.commit_snapshot(MemorySnapshot(1, (new,)), expected_revision=1, event_id="new", ts="now",
        session_key="cli:test", history_text="new", stage_publish=True,
        publish_expected_text=old, publish_target_text=render_memory_markdown((new,)))
    before = bytes_before(source)
    assert not inspect_workspace(source)["report"]["source_choice_required"]
    await convert_workspace(source, tmp_path / "recovered", config=config())
    assert inspect_workspace(tmp_path / "recovered")["records"] == (new,)
    assert bytes_before(source) == before


@pytest.mark.asyncio
async def test_markdown_only_preserves_history_without_model(source, tmp_path):
    # A supported six-section Markdown workspace without a database.
    markdown_only = tmp_path / "markdown"
    (markdown_only / "memory").mkdir(parents=True)
    (markdown_only / "memory/MEMORY.md").write_text(render_memory_markdown(()))
    (markdown_only / "memory/HISTORY.md").write_text("Original history, unchanged.")
    await convert_workspace(markdown_only, tmp_path / "imported", config=config())
    events = inspect_workspace(tmp_path / "imported")["memory"]["raw_events"]
    assert events[0]["plain_text"] == "Original history, unchanged."


@pytest.mark.asyncio
async def test_existing_destination_is_never_overwritten(source, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_text("important")
    with pytest.raises(ValueError, match="Destination"):
        await convert_workspace(source, destination, config=config())
    assert (destination / "keep").read_text() == "important"


@pytest.mark.asyncio
async def test_storage_symlinks_cannot_write_outside_copy(source, tmp_path):
    external = tmp_path / "external.md"
    external.write_text(render_memory_markdown(()))
    (source / "memory/MEMORY.md").unlink()
    (source / "memory/MEMORY.md").symlink_to(external)
    before = external.read_bytes()
    with pytest.raises(ValueError, match="symlinks"):
        await convert_workspace(source, tmp_path / "linked", config=config(), memory_source="database")
    assert external.read_bytes() == before
    assert not (tmp_path / "linked").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["upgrade-report.json", "upgrade-originals"])
@pytest.mark.parametrize("exists", [False, True])
async def test_upgrade_output_symlinks_cannot_write_outside_copy(source, tmp_path, name, exists):
    external = tmp_path / "external"
    if exists:
        if name == "upgrade-originals":
            external.mkdir()
            (external / "keep").write_text("original")
        else:
            external.write_text("original")
    (source / name).symlink_to(external)
    before = bytes_before(source)
    with pytest.raises(ValueError, match="symlinks"):
        await convert_workspace(source, tmp_path / "linked", config=config())
    assert bytes_before(source) == before
    assert not (tmp_path / "linked").exists()
    if not exists:
        assert not external.exists()
    elif name == "upgrade-originals":
        assert list(external.iterdir()) == [external / "keep"]
        assert (external / "keep").read_text() == "original"
    else:
        assert external.read_text() == "original"
