from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.memory_db import MemoryDatabase, MemoryRecord, MemorySnapshot
from nanobot.agent.memory_service import MemoryService
from nanobot.agent.memory_sync import (
    MarkdownValidationError,
    MemoryConflictError,
    MemorySynchronizer,
    MemorySyncStatus,
    diff_memory_snapshots,
    parse_memory_markdown,
    render_memory_markdown,
)
from nanobot.session.manager import SessionManager


def _provider() -> AsyncMock:
    return AsyncMock()


def _replace_empty_section(content: str, main_class: str, replacement: str) -> str:
    placeholders = {
        "personal_profile": "(Stable background information about the user)",
        "preferences": "(How the user prefers to communicate and collaborate)",
        "constraints": "(Rules, boundaries, and requirements that must be respected)",
        "projects": "(Information about ongoing learning and work projects)",
        "daily_life": "(Daily routines, interests, and hobbies)",
        "plans_commitments": "(Future plans, commitments, deadlines, and to-dos)",
    }
    return content.replace(placeholders[main_class], replacement)


def _seed(db: MemoryDatabase, records: tuple[MemoryRecord, ...]) -> int:
    return db.commit_snapshot(
        MemorySnapshot(0, records),
        expected_revision=0,
        event_id="seed",
        ts="2026-09-05T00:00:00",
        session_key="test",
        history_text="seed",
    )


def test_strict_parser_round_trips_all_six_sections_and_empty_sections() -> None:
    records = (
        MemoryRecord.create("personal_profile", "location", "Lives in Boston"),
        MemoryRecord.create("projects", "active", "Builds nanobot"),
    )
    rendered = render_memory_markdown(records)

    parsed = parse_memory_markdown(rendered)

    assert {(item.main_class, item.sub_class, item.text) for item in parsed} == {
        ("personal_profile", "location", "Lives in Boston"),
        ("projects", "active", "Builds nanobot"),
    }
    assert len([line for line in rendered.splitlines() if line.startswith("## ")]) == 6


@pytest.mark.parametrize(
    ("mutate", "line_number", "message"),
    [
        (lambda text: text.replace("## Projects", "## Unknown"), 17, "unknown memory category"),
        (
            lambda text: _replace_empty_section(text, "preferences", "- not parseable"),
            11,
            "expected '- sub_class: text'",
        ),
        (
            lambda text: _replace_empty_section(text, "preferences", "- : value"),
            11,
            "sub_class must not be empty",
        ),
    ],
)
def test_strict_parser_reports_first_invalid_line(mutate, line_number, message) -> None:
    with pytest.raises(MarkdownValidationError) as caught:
        parse_memory_markdown(mutate(render_memory_markdown(())))
    assert caught.value.line_number == line_number
    assert message in str(caught.value)


def test_empty_file_and_missing_category_are_invalid() -> None:
    with pytest.raises(MarkdownValidationError, match="line 1"):
        parse_memory_markdown("")
    incomplete = render_memory_markdown(()).replace(
        "## Plans and Commitments\n\n(Future plans, commitments, deadlines, and to-dos)\n\n",
        "",
    )
    with pytest.raises(MarkdownValidationError, match="missing memory categories"):
        parse_memory_markdown(incomplete)


def test_snapshot_diff_preserves_unchanged_ids_and_represents_edits_as_add_remove() -> None:
    keep = MemoryRecord("keep-custom-id", "projects", "active", "Keep this fact")
    modify = MemoryRecord.create("preferences", "style", "Old style")
    move = MemoryRecord.create("projects", "topic", "Move me")
    sibling = MemoryRecord.create("projects", "active", "Second fact in same subclass")
    current = MemorySnapshot(4, (keep, modify, move, sibling))
    parsed = (
        MemoryRecord.create("projects", "active", "Keep this fact"),
        MemoryRecord.create("preferences", "style", "New style"),
        MemoryRecord.create("constraints", "topic", "Move me"),
        MemoryRecord.create("projects", "active", "Second fact in same subclass"),
    )

    target, changes = diff_memory_snapshots(current, parsed)

    assert target.revision == 4
    assert {item.memory_id for item in changes.unchanged} == {"keep-custom-id", sibling.memory_id}
    assert {(item.main_class, item.text) for item in changes.removed} == {
        ("preferences", "Old style"),
        ("projects", "Move me"),
    }
    assert {(item.main_class, item.text) for item in changes.added} == {
        ("preferences", "New style"),
        ("constraints", "Move me"),
    }
    assert sum(item.sub_class == "active" for item in target.memories) == 2


@pytest.mark.asyncio
async def test_manual_core_and_dynamic_edits_apply_on_next_context(tmp_path) -> None:
    service = MemoryService(tmp_path, _provider(), "test-model")
    await service.sync_markdown()
    original = service.database.memory_file.read_text(encoding="utf-8")
    edited = _replace_empty_section(original, "preferences", "- style: Use detailed answers")
    edited = _replace_empty_section(
        edited, "projects", "- active: Markdown synchronization project"
    )
    service.database.memory_file.write_text(edited, encoding="utf-8")

    context = await service.prepare_context("synchronization", retrieval_budget=5)

    assert [item.text for item in context.core_items] == ["Use detailed answers"]
    assert [item.text for item in context.retrieved_items] == ["Markdown synchronization project"]
    assert service.database.read_snapshot().revision == 1
    assert service.synchronizer.read_state().published_revision == 1
    assert service.database.memory_file.read_text(encoding="utf-8") == render_memory_markdown(
        service.database.read_snapshot().memories
    )


@pytest.mark.asyncio
async def test_add_delete_subclass_change_and_category_move_commit_exact_snapshot(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.initialize()
    original = (
        MemoryRecord.create("preferences", "style", "Concise"),
        MemoryRecord.create("projects", "active", "Alpha"),
        MemoryRecord.create("projects", "active", "Beta"),
    )
    _seed(db, original)
    sync = MemorySynchronizer(db)
    sync.sync()
    edited_records = (
        MemoryRecord.create("preferences", "detail", "Concise"),
        MemoryRecord.create("constraints", "active", "Alpha"),
        MemoryRecord.create("daily_life", "hobby", "Cycling"),
    )
    db.memory_file.write_text(render_memory_markdown(edited_records), encoding="utf-8")
    service = MemoryService(tmp_path, _provider(), "test-model", database=db)

    result = await service.sync_markdown()

    assert result.status == MemorySyncStatus.IMPORTED
    assert db.read_snapshot().revision == 2
    assert {
        (item.main_class, item.sub_class, item.text) for item in db.read_snapshot().memories
    } == {(item.main_class, item.sub_class, item.text) for item in edited_records}
    assert len(result.changes.added) == 3
    assert len(result.changes.removed) == 3
    assert db.query_dynamic_memories("Beta") == ()
    assert [item.text for item in db.query_dynamic_memories("Cycling")] == ["Cycling"]


@pytest.mark.asyncio
async def test_invalid_edit_is_rejected_as_one_unit_and_file_is_preserved(tmp_path) -> None:
    service = MemoryService(tmp_path, _provider(), "test-model")
    await service.sync_markdown()
    before = service.database.read_snapshot()
    invalid = _replace_empty_section(
        service.database.memory_file.read_text(encoding="utf-8"),
        "preferences",
        "- style: Valid entry\n- invalid entry",
    )
    service.database.memory_file.write_text(invalid, encoding="utf-8")

    with pytest.raises(MarkdownValidationError, match="expected '- sub_class: text'"):
        await service.prepare_context("anything")

    assert service.database.read_snapshot() == before
    assert service.database.list_raw_events() == []
    assert service.database.memory_file.read_text(encoding="utf-8") == invalid


@pytest.mark.asyncio
async def test_repeated_sync_is_revision_stable_and_valid_empty_template_clears(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.initialize()
    _seed(db, (MemoryRecord.create("preferences", "style", "Concise"),))
    service = MemoryService(tmp_path, _provider(), "test-model", database=db)
    await service.sync_markdown()
    revision = db.read_snapshot().revision

    await service.sync_markdown()
    await service.prepare_context("")
    assert db.read_snapshot().revision == revision

    db.memory_file.write_text(render_memory_markdown(()), encoding="utf-8")
    result = await service.sync_markdown()
    assert result.status == MemorySyncStatus.IMPORTED
    assert db.read_snapshot() == MemorySnapshot(revision + 1)


@pytest.mark.asyncio
async def test_missing_view_is_rebuilt_but_empty_file_is_not_a_clear_command(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.initialize()
    record = MemoryRecord.create("preferences", "style", "Concise")
    _seed(db, (record,))
    service = MemoryService(tmp_path, _provider(), "test-model", database=db)

    await service.prepare_context("")
    assert "Concise" in db.memory_file.read_text(encoding="utf-8")
    db.memory_file.write_text("", encoding="utf-8")

    with pytest.raises(MarkdownValidationError, match="empty files"):
        await service.prepare_context("")
    assert db.read_snapshot().memories == (record,)
    assert db.memory_file.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_edit_during_extraction_is_imported_and_stale_model_snapshot_is_discarded(
    tmp_path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class WaitingPipeline:
        async def extract_snapshot(self, messages, current):
            started.set()
            await release.wait()
            return (
                MemorySnapshot(
                    current.revision,
                    (MemoryRecord.create("preferences", "style", "Model overwrite"),),
                ),
                "model extraction",
            )

    service = MemoryService(
        tmp_path,
        _provider(),
        "test-model",
        pipeline=WaitingPipeline(),  # type: ignore[arg-type]
    )
    await service.sync_markdown()
    task = asyncio.create_task(
        service.consolidate([{"role": "user", "content": "remember"}], session_key="cli:test")
    )
    await started.wait()
    edited = _replace_empty_section(
        service.database.memory_file.read_text(encoding="utf-8"),
        "preferences",
        "- style: Human edit wins",
    )
    service.database.memory_file.write_text(edited, encoding="utf-8")
    release.set()

    result = await task

    assert not result.database_committed
    assert not result.retryable
    assert "stale extraction result was discarded" in (result.error or "")
    assert [item.text for item in service.database.read_snapshot().memories] == ["Human edit wins"]
    assert service.database.read_snapshot().revision == 1
    assert [event["candidate_type"] for event in service.database.list_raw_events()] == [
        "markdown_edit"
    ]

    consolidator = MemoryConsolidator(
        service,
        _provider(),
        "test-model",
        SessionManager(tmp_path),
        4096,
        MagicMock(return_value=[]),
        MagicMock(return_value=[]),
    )
    service.consolidate = AsyncMock(return_value=result)  # type: ignore[method-assign]
    archive_result = await consolidator.archive_messages(
        [{"role": "user", "content": "do not retry"}], session_key="cli:test"
    )
    assert archive_result is result
    service.consolidate.assert_awaited_once()


def test_pending_publish_recovers_when_file_is_still_expected_old_text(
    tmp_path, monkeypatch
) -> None:
    service = MemoryService(tmp_path, _provider(), "test-model")
    service.synchronizer.sync()
    old_text = service.database.memory_file.read_text(encoding="utf-8")
    target = (MemoryRecord.create("preferences", "style", "Recovered"),)
    service.database.commit_snapshot(
        MemorySnapshot(0, target),
        expected_revision=0,
        event_id="commit",
        ts="2026-09-05T01:00:00",
        session_key="test",
        history_text="commit",
        publish_expected_text=old_text,
        publish_target_text=render_memory_markdown(target),
        stage_publish=True,
    )

    recovered = MemoryService(tmp_path, _provider(), "test-model", database=service.database)

    assert "Recovered" in recovered.database.memory_file.read_text(encoding="utf-8")
    assert recovered.synchronizer.read_state().pending_revision is None
    assert recovered.synchronizer.read_state().published_revision == 1


def test_pending_publish_confirms_when_file_was_replaced_before_interruption(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.initialize()
    sync = MemorySynchronizer(db)
    sync.sync()
    old_text = db.memory_file.read_text(encoding="utf-8")
    target = (MemoryRecord.create("preferences", "style", "Already replaced"),)
    target_text = render_memory_markdown(target)
    db.commit_snapshot(
        MemorySnapshot(0, target),
        expected_revision=0,
        event_id="commit",
        ts="2026-09-05T01:00:00",
        session_key="test",
        history_text="commit",
        publish_expected_text=old_text,
        publish_target_text=target_text,
        stage_publish=True,
    )
    db.memory_file.write_text(target_text, encoding="utf-8")

    recovered = MemoryService(tmp_path, _provider(), "test-model", database=db)

    assert recovered.synchronizer.read_state().pending_revision is None
    assert recovered.synchronizer.read_state().published_text == target_text


def test_pending_publish_conflict_preserves_human_file_and_database_target(tmp_path) -> None:
    db = MemoryDatabase(tmp_path)
    db.initialize()
    sync = MemorySynchronizer(db)
    sync.sync()
    old_text = db.memory_file.read_text(encoding="utf-8")
    database_target = (MemoryRecord.create("preferences", "style", "Database target"),)
    db.commit_snapshot(
        MemorySnapshot(0, database_target),
        expected_revision=0,
        event_id="commit",
        ts="2026-09-05T01:00:00",
        session_key="test",
        history_text="commit",
        publish_expected_text=old_text,
        publish_target_text=render_memory_markdown(database_target),
        stage_publish=True,
    )
    human_text = _replace_empty_section(old_text, "preferences", "- style: Human concurrent edit")
    db.memory_file.write_text(human_text, encoding="utf-8")

    with pytest.raises(MemoryConflictError, match="file was preserved"):
        MemoryService(tmp_path, _provider(), "test-model", database=db)

    assert db.memory_file.read_text(encoding="utf-8") == human_text
    assert [item.text for item in db.read_snapshot().memories] == ["Database target"]
    assert db.read_publish_state()["pending_revision"] == 1
