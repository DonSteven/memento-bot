# Phase 2 test content:
# - verifies v2 mode writes the structured pipeline results into the root memory files
# - verifies v2 and legacy can render equivalent user-visible MEMORY/HISTORY outputs
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_2/agent/test_memory_shadow_mode.py

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "Remember I prefer concise replies.", "timestamp": "2026-04-01T10:00:00"},
        {"role": "assistant", "content": "Noted.", "timestamp": "2026-04-01T10:00:01"},
    ]


def _legacy_response() -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="legacy_call_1",
                name="save_memory",
                arguments={
                    "history_entry": "[2026-04-01 10:00] User prefers concise replies.",
                    "memory_update": (
                        "# Long-term Memory\n\n"
                        "This file stores important information that should persist across sessions.\n\n"
                        "## Personal Profile\n\n"
                        "(Stable background information about the user)\n\n"
                        "## Preferences\n\n"
                        "- reply_style: User prefers concise replies.\n\n"
                        "## Constraints\n\n"
                        "(Rules, boundaries, and requirements that must be respected)\n\n"
                        "## Projects\n\n"
                        "(Information about ongoing learning and work projects)\n\n"
                        "## Daily Life\n\n"
                        "(Daily routines, interests, and hobbies)\n\n"
                        "## Plans and Commitments\n\n"
                        "(Future plans, commitments, deadlines, and to-dos)\n\n"
                        "---\n\n"
                        "*This file is automatically updated by nanobot when important information should be remembered.*"
                    ),
                },
            )
        ],
    )


def _v2_response() -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="v2_call_1",
                name="save_memory_structured",
                arguments={
                    "history_entry": "[2026-04-01 10:00] User prefers concise replies.",
                    "canonical_memories": [
                        {
                            "main_class": "preferences",
                            "sub_class": "reply_style",
                            "text": "User prefers concise replies.",
                            "status": "active",
                        }
                    ],
                },
            )
        ],
    )


@pytest.mark.asyncio
async def test_v2_mode_writes_root_outputs_without_sidecar_dir(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, mode="v2")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(return_value=_v2_response())

    result = await store.consolidate(_messages(), provider, "test-model")

    assert result is True
    assert "User prefers concise replies." in store.memory_file.read_text(encoding="utf-8")
    assert "User prefers concise replies." in store.history_file.read_text(encoding="utf-8")
    assert not (tmp_path / "memory" / "shadow").exists()


@pytest.mark.asyncio
async def test_v2_mode_matches_legacy_user_visible_outputs_exactly(tmp_path: Path) -> None:
    legacy_store = MemoryStore(tmp_path / "legacy", mode="legacy")
    v2_store = MemoryStore(tmp_path / "v2", mode="v2")

    legacy_provider = AsyncMock()
    legacy_provider.chat_with_retry = AsyncMock(return_value=_legacy_response())

    v2_provider = AsyncMock()
    v2_provider.chat_with_retry = AsyncMock(return_value=_v2_response())

    legacy_result = await legacy_store.consolidate(_messages(), legacy_provider, "test-model")
    v2_result = await v2_store.consolidate(_messages(), v2_provider, "test-model")

    assert legacy_result is True
    assert v2_result is True
    assert legacy_store.memory_file.read_text(encoding="utf-8") == v2_store.memory_file.read_text(encoding="utf-8")
    assert legacy_store.history_file.read_text(encoding="utf-8") == v2_store.history_file.read_text(encoding="utf-8")
