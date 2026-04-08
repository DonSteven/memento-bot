# Phase 0 test content:
# - provides a fixture-driven legacy baseline for consolidation boundary selection
# - gives later v2/fts ablation runs a stable comparison sample
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_0/agent/test_memory_replay_eval.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import nanobot.agent.memory as memory_module
from nanobot.agent.memory import MemoryConsolidator
from nanobot.session.manager import Session, SessionManager


def _load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "memory" / f"{name}.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.generation.max_tokens = 0
    provider.get_default_model.return_value = "test-model"
    return provider


def test_phase0_legacy_boundary_fixture(tmp_path: Path, monkeypatch) -> None:
    fixture = _load_fixture("phase0_boundary")
    session = Session(key=fixture["session_key"], messages=fixture["messages"])

    consolidator = MemoryConsolidator(
        workspace=tmp_path,
        provider=_make_provider(),
        model="test-model",
        sessions=SessionManager(tmp_path),
        context_window_tokens=4096,
        build_messages=lambda **_kwargs: [],
        get_tool_definitions=lambda: [],
        mode=fixture["memory_mode"],
    )

    token_map = fixture["token_map"]
    monkeypatch.setattr(
        memory_module,
        "estimate_message_tokens",
        lambda message: token_map[message["content"]],
    )

    boundary = consolidator.pick_consolidation_boundary(
        session,
        fixture["tokens_to_remove"],
    )

    assert boundary == (
        fixture["expected"]["boundary_index"],
        fixture["expected"]["removed_tokens"],
    )

    end_idx = fixture["expected"]["boundary_index"]
    archived = session.messages[session.last_consolidated:end_idx]
    assert [message["content"] for message in archived] == fixture["expected"]["archived_contents"]
