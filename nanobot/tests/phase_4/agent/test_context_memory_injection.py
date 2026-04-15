# Phase 4 test content:
# - verifies ContextBuilder keeps full memory injection for legacy mode
# - verifies v2 mode injects core memory plus relevant retrieved snippets
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

    builder = ContextBuilder(workspace, memory_mode="legacy")
    messages = builder.build_messages(history=[], current_message="How should you answer?")

    system_prompt = messages[0]["content"]
    assert "# Memory" in system_prompt
    assert "reply_style: User prefers concise answers." in system_prompt


def test_context_builder_v2_injects_core_memory_without_full_view_fallback(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    db = MemoryDatabase(workspace)
    db.upsert_canonical_memory(
        memory_id="mem_profile",
        main_class="personal_profile",
        sub_class="environment",
        text="User works on Linux.",
    )
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

    builder = ContextBuilder(workspace, memory_mode="v2")
    messages = builder.build_messages(history=[], current_message="   ")

    system_prompt = messages[0]["content"]
    assert "# Memory" in system_prompt
    assert "## Core Memory" in system_prompt
    assert "[personal_profile/environment] User works on Linux." in system_prompt
    assert "[preferences/reply_style] User prefers concise answers." in system_prompt
    assert "## Retrieved Memory" not in system_prompt
    assert "The active project is nanobot." not in system_prompt


def test_context_builder_v2_injects_core_memory_and_filters_irrelevant_matches(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    db = MemoryDatabase(workspace)
    db.upsert_canonical_memory(
        memory_id="mem_profile",
        main_class="personal_profile",
        sub_class="environment",
        text="User works on Linux.",
    )
    db.upsert_canonical_memory(
        memory_id="mem_pref",
        main_class="preferences",
        sub_class="reply_style",
        text="User prefers concise answers.",
    )
    db.upsert_canonical_memory(
        memory_id="mem_constraint",
        main_class="constraints",
        sub_class="tooling",
        text="Avoid interactive git commands.",
    )
    db.upsert_canonical_memory(
        memory_id="mem_project",
        main_class="projects",
        sub_class="active_project",
        text="The active project is nanobot.",
    )
    db.write_views()

    builder = ContextBuilder(workspace, memory_mode="v2")
    messages = builder.build_messages(history=[], current_message="Please answer concisely.")

    system_prompt = messages[0]["content"]
    assert "## Core Memory" in system_prompt
    assert "[personal_profile/environment] User works on Linux." in system_prompt
    assert "[preferences/reply_style] User prefers concise answers." in system_prompt
    assert "[constraints/tooling] Avoid interactive git commands." in system_prompt
    assert "The active project is nanobot." not in system_prompt


def test_context_builder_v2_appends_relevant_project_hits_after_core_memory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    db = MemoryDatabase(workspace)
    db.upsert_canonical_memory(
        memory_id="mem_profile",
        main_class="personal_profile",
        sub_class="environment",
        text="User works on Linux.",
    )
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

    builder = ContextBuilder(workspace, memory_mode="v2")
    messages = builder.build_messages(history=[], current_message="What's the active project?")

    system_prompt = messages[0]["content"]
    assert "## Core Memory" in system_prompt
    assert "## Retrieved Memory" in system_prompt
    assert "[projects/active_project] The active project is nanobot." in system_prompt


def test_context_builder_with_web_knowledge_enabled_adds_kb_search_guidance(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    builder = ContextBuilder(workspace, knowledge_enabled=True)

    messages = builder.build_messages(history=[], current_message="What changed in Linux 6.9?")

    system_prompt = messages[0]["content"]
    assert "use 'kb_search' before calling 'web_search' or 'web_fetch'" in system_prompt
