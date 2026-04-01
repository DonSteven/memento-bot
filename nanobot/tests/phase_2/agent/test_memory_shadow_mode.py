# Phase 2 test content:
# - verifies shadow mode keeps legacy MEMORY/HISTORY writes intact while writing sidecar shadow artifacts
# - verifies shadow pipeline failures do not break the user-visible legacy consolidation result
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


@pytest.mark.asyncio
async def test_shadow_mode_preserves_legacy_outputs_and_writes_sidecar(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, mode="shadow")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="legacy_call_1",
                    name="save_memory",
                    arguments={
                        "history_entry": "[2026-04-01 10:00] User prefers concise replies.",
                        "memory_update": "# Long-term Memory\n\n## Preferences\n\n- User prefers concise replies.",
                    },
                )
            ],
        ),
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="shadow_call_1",
                    name="save_memory_shadow",
                    arguments={
                        "history_entry": "[2026-04-01 10:00] User prefers concise replies.",
                        "canonical_memories": [
                            {
                                "main_class": "preferences",
                                "sub_class": "reply_style",
                                "text": "User prefers concise replies.",
                                "confidence": 0.92,
                            }
                        ],
                    },
                )
            ],
        ),
    ])

    result = await store.consolidate(_messages(), provider, "test-model")

    assert result is True
    assert "User prefers concise replies." in store.memory_file.read_text(encoding="utf-8")
    assert "User prefers concise replies." in store.history_file.read_text(encoding="utf-8")
    shadow_dir = tmp_path / "memory" / "shadow"
    assert (shadow_dir / "MEMORY.md").exists()
    assert (shadow_dir / "HISTORY.md").exists()
    assert (shadow_dir / "last_payload.json").exists()


@pytest.mark.asyncio
async def test_shadow_pipeline_failure_does_not_break_legacy_consolidation(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, mode="shadow")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="legacy_call_1",
                    name="save_memory",
                    arguments={
                        "history_entry": "[2026-04-01 10:00] User prefers concise replies.",
                        "memory_update": "# Long-term Memory\n\n## Preferences\n\n- User prefers concise replies.",
                    },
                )
            ],
        )
    )
    store.shadow_pipeline.ingest_chunk = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[union-attr]

    result = await store.consolidate(_messages(), provider, "test-model")

    assert result is True
    assert "User prefers concise replies." in store.memory_file.read_text(encoding="utf-8")
    assert "User prefers concise replies." in store.history_file.read_text(encoding="utf-8")
