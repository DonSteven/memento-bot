# Knowledge test content:
# - verifies ContextBuilder adds kb_search guidance when web knowledge is enabled
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/knowledge/test_knowledge_context.py

from __future__ import annotations

from nanobot.agent.context import ContextBuilder


def test_context_builder_with_web_knowledge_enabled_adds_kb_search_guidance(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    builder = ContextBuilder(workspace, knowledge_enabled=True)

    messages = builder.build_messages(history=[], current_message="What changed in Linux 6.9?")

    system_prompt = messages[0]["content"]
    assert "use 'kb_search' before calling 'web_search' or 'web_fetch'" in system_prompt
