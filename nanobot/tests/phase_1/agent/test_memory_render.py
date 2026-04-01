# Phase 1 test content:
# - verifies the legacy MEMORY.md and HISTORY.md views can be rendered from the new storage base
# - verifies placeholder sections remain stable when a section has no canonical memories
# - verifies write_views emits the rendered markdown into workspace memory files
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
