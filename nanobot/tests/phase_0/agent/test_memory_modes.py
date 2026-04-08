# Phase 0 test content:
# - verifies the new memory.mode config defaults to legacy
# - verifies AgentLoop and Nanobot.from_config propagate memory_mode into the memory layer
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_0/agent/test_memory_modes.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.nanobot import Nanobot


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 1024
    return provider


def _write_config(tmp_path: Path, *, memory_mode: str) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {"defaults": {"model": "openai/gpt-4.1"}},
        "memory": {"mode": memory_mode},
    }))
    return config_path


def test_config_defaults_to_legacy_memory_mode() -> None:
    config = Config()
    assert config.memory.mode == "legacy"


def test_config_rejects_shadow_memory_mode() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"memory": {"mode": "shadow"}})


def test_config_rejects_retrieval_mode() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"memory": {"mode": "v2", "retrieval_mode": "fts"}})


def test_agent_loop_propagates_memory_mode(tmp_path: Path) -> None:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=_make_provider(),
        workspace=tmp_path,
        memory_mode="v2",
    )

    assert loop.memory_mode == "v2"
    assert loop.memory_consolidator.mode == "v2"
    assert loop.memory_consolidator.store.mode == "v2"


def test_nanobot_from_config_passes_memory_mode(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, memory_mode="v2")

    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    assert bot._loop.memory_mode == "v2"
    assert bot._loop.memory_consolidator.mode == "v2"
