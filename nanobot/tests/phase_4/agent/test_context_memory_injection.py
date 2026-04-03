# Phase 4 test content:
# - verifies ContextBuilder keeps full_view injection as the default behavior
# - verifies fts mode injects only retrieved memory snippets for the current query in v2 mode
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_4/agent/test_context_memory_injection.py

from __future__ import annotations

from pathlib import Path

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory_db import MemoryDatabase


def test_context_builder_full_view_injects_full_memory_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    (workspace / "memory" / "MEMORY.md").write_text(
        "# Long-term Memory\n\n## Preferences\n\n- reply_style: User prefers concise answers.",
        encoding="utf-8",
    )

    builder = ContextBuilder(workspace, memory_mode="legacy", retrieval_mode="full_view")
    messages = builder.build_messages(history=[], current_message="How should you answer?")

    system_prompt = messages[0]["content"]
    assert "# Memory" in system_prompt
    assert "reply_style: User prefers concise answers." in system_prompt


def test_context_builder_fts_injects_only_retrieved_memory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    db = MemoryDatabase(workspace)
    db.upsert_canonical_memory(
        memory_id="mem_pref",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise answers.",
    )
    db.upsert_canonical_memory(
        memory_id="mem_project",
        main_class="projects",
        sub_class="active_project",
        text="The active project is nanobot.",
    )
    db.write_views()

    builder = ContextBuilder(workspace, memory_mode="v2", retrieval_mode="fts")
    messages = builder.build_messages(history=[], current_message="Please answer concisely.")

    system_prompt = messages[0]["content"]
    assert "## Retrieved Memory" in system_prompt
    assert "[preferences/reply_style] User prefers concise answers." in system_prompt
    assert "The active project is nanobot." not in system_prompt
