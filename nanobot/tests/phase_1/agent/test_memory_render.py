# Phase 1 test content:
# - verifies the legacy MEMORY.md and HISTORY.md views can be rendered from the new storage base
# - verifies placeholder sections remain stable when a section has no canonical memories
# - verifies write_views emits the rendered markdown into workspace memory files
# - verifies markdown rendering remains stable for empty text and sorting edge cases
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_1/agent/test_memory_render.py

from __future__ import annotations

import json
from pathlib import Path

from nanobot.agent.memory_db import MemoryDatabase, render_history_markdown, render_memory_markdown


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

    for event in fixture["raw_events"]:
        db.insert_raw_event(
            event_id=event["event_id"],
            ts=event["ts"],
            session_key=event["session_key"],
            history_text=event["history_text"],
            plain_text=event.get("plain_text", ""),
            main_class=event.get("main_class"),
            sub_class=event.get("sub_class"),
            candidate_type=event.get("candidate_type"),
            confidence=event.get("confidence", 0.0),
        )
    for memory in fixture["canonical_memories"]:
        db.upsert_canonical_memory(
            memory_id=memory["memory_id"],
            main_class=memory["main_class"],
            sub_class=memory.get("sub_class", ""),
            text=memory["text"],
            confidence=memory.get("confidence", 0.0),
            priority=memory.get("priority", 0.0),
            status=memory.get("status", "active"),
            version=memory.get("version", 1),
        )

    db.write_views()

    assert db.memory_file.read_text(encoding="utf-8") == fixture["expected_memory_markdown"]
    assert db.history_file.read_text(encoding="utf-8") == fixture["expected_history_markdown"]


def test_render_memory_markdown_all_sections_empty_uses_placeholders() -> None:
    rendered = render_memory_markdown([])

    assert "## User Information\n\n(Important facts about the user)" in rendered
    assert "## Preferences\n\n(User preferences learned over time)" in rendered
    assert "## Project Context\n\n(Information about ongoing projects)" in rendered
    assert "## Important Notes\n\n(Things to remember)" in rendered


def test_render_memory_markdown_ignores_empty_text_items() -> None:
    rendered = render_memory_markdown([
        {
            "memory_id": "mem_empty",
            "main_class": "preferences",
            "sub_class": "reply_style",
            "text": "   ",
        }
    ])

    assert "## Preferences\n\n(User preferences learned over time)" in rendered
    assert "- reply_style:" not in rendered


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


def test_render_history_markdown_filters_empty_entries_and_keeps_order() -> None:
    rendered = render_history_markdown([
        {"event_id": "evt_2", "ts": "2026-04-01T09:05:00", "history_text": "  "},
        {"event_id": "evt_3", "ts": "2026-04-01T09:06:00", "history_text": "[2026-04-01 09:06] Third."},
        {"event_id": "evt_1", "ts": "2026-04-01T09:04:00", "history_text": "[2026-04-01 09:04] First."},
    ])

    assert rendered == "[2026-04-01 09:04] First.\n\n[2026-04-01 09:06] Third."
