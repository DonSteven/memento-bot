# Phase 1 test content:
# - verifies the legacy MEMORY.md and HISTORY.md views can be rendered from the new storage base
# - verifies placeholder sections remain stable when a section has no canonical memories
# - verifies the synchronizer emits rendered Markdown into workspace memory files
# - verifies markdown rendering remains stable for empty text and sorting edge cases
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_1/agent/test_memory_render.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agent.memory_db import (
    MemoryDatabase,
    MemoryRecord,
    MemorySnapshot,
    render_history_markdown,
)
from nanobot.agent.memory_sync import (
    MarkdownValidationError,
    MemorySynchronizer,
    parse_memory_markdown,
    render_memory_markdown,
)


def _load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "memory" / f"{name}.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_render_memory_markdown_from_fixture() -> None:
    fixture = _load_fixture("render_views")

    rendered = render_memory_markdown(fixture["canonical_memories"])

    assert rendered == fixture["expected_memory_markdown"]


def test_render_history_markdown_from_fixture() -> None:
    fixture = _load_fixture("render_views")

    rendered = render_history_markdown(fixture["raw_events"])

    assert rendered == fixture["expected_history_markdown"]


def test_write_views_writes_memory_and_history_files(tmp_path) -> None:
    fixture = _load_fixture("render_views")
    db = MemoryDatabase(tmp_path)

    db.initialize()
    records = tuple(
        MemoryRecord(
            memory["memory_id"], memory["main_class"], memory["sub_class"], memory["text"]
        )
        for memory in fixture["canonical_memories"]
    )
    revision = 0
    for event in fixture["raw_events"]:
        revision = db.commit_snapshot(
            MemorySnapshot(revision, records),
            expected_revision=revision,
            event_id=event["event_id"],
            ts=event["ts"],
            session_key=event["session_key"],
            history_text=event["history_text"],
            plain_text=event.get("plain_text", ""),
            candidate_type=event.get("candidate_type") or "fixture",
        )

    MemorySynchronizer(db).sync()

    assert db.memory_file.read_text(encoding="utf-8") == fixture["expected_memory_markdown"]
    assert db.history_file.read_text(encoding="utf-8") == fixture["expected_history_markdown"]


def test_render_memory_markdown_all_sections_empty_uses_placeholders() -> None:
    rendered = render_memory_markdown([])

    assert "## Personal Profile\n\n(Stable background information about the user)" in rendered
    assert "## Preferences\n\n(How the user prefers to communicate and collaborate)" in rendered
    assert "## Constraints\n\n(Rules, boundaries, and requirements that must be respected)" in rendered
    assert "## Projects\n\n(Information about ongoing learning and work projects)" in rendered
    assert "## Daily Life\n\n(Daily routines, interests, and hobbies)" in rendered
    assert "## Plans and Commitments\n\n(Future plans, commitments, deadlines, and to-dos)" in rendered


def test_render_memory_markdown_rejects_empty_text_items() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        render_memory_markdown([
            {
                "memory_id": "mem_empty",
                "main_class": "preferences",
                "sub_class": "reply_style",
                "text": "   ",
            }
        ])


def test_render_memory_markdown_rejects_blank_subclass_items() -> None:
    with pytest.raises(ValueError, match="sub_class"):
        render_memory_markdown([
            {
                "memory_id": "mem_blank_subclass",
                "main_class": "preferences",
                "sub_class": "   ",
                "text": "User prefers concise answers.",
            }
        ])


def test_render_memory_markdown_sorts_stably_by_subclass_then_memory_id() -> None:
    rendered = render_memory_markdown([
        {
            "memory_id": "mem_b",
            "main_class": "preferences",
            "sub_class": "z_style",
            "text": "Later subclass item.",
        },
        {
            "memory_id": "mem_c",
            "main_class": "preferences",
            "sub_class": "a_style",
            "text": "Second item in same subclass.",
        },
        {
            "memory_id": "mem_a",
            "main_class": "preferences",
            "sub_class": "a_style",
            "text": "First item in same subclass.",
        },
    ])

    first = rendered.index("- a_style: First item in same subclass.")
    second = rendered.index("- a_style: Second item in same subclass.")
    third = rendered.index("- z_style: Later subclass item.")
    assert first < second < third


def test_parse_memory_markdown_rejects_legacy_bullets_without_subclass() -> None:
    content = render_memory_markdown([]).replace(
        "(How the user prefers to communicate and collaborate)",
        "- User prefers concise answers.",
    )
    with pytest.raises(MarkdownValidationError, match="line 11"):
        parse_memory_markdown(content)


def test_render_history_markdown_filters_empty_entries_and_keeps_order() -> None:
    rendered = render_history_markdown([
        {"event_id": "evt_2", "ts": "2026-04-01T09:05:00", "history_text": "  "},
        {"event_id": "evt_3", "ts": "2026-04-01T09:06:00", "history_text": "[2026-04-01 09:06] Third."},
        {"event_id": "evt_1", "ts": "2026-04-01T09:04:00", "history_text": "[2026-04-01 09:04] First."},
    ])

    assert rendered == "[2026-04-01 09:04] First.\n\n[2026-04-01 09:06] Third.\n\n"
