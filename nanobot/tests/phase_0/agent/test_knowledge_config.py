# Phase 0 test content:
# - verifies external web knowledge config defaults to disabled
# - verifies Nanobot.from_config preserves the configured knowledge settings
# - verifies AgentLoop wires kb_search and system hooks when knowledge is enabled
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_0/agent/test_knowledge_config.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, KnowledgeConfig
from nanobot.nanobot import Nanobot


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 1024
    return provider


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {"defaults": {"model": "openai/gpt-4.1"}},
        "knowledge": {
            "enabled": False,
            "embeddingModel": "sentence-transformers/test-model",
            "docLimit": 6,
        },
    }))
    return config_path


def test_config_defaults_to_disabled_web_knowledge() -> None:
    config = Config()
    assert config.knowledge.enabled is False


def test_nanobot_from_config_passes_knowledge_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    assert bot._loop.knowledge_config.enabled is False
    assert bot._loop.knowledge_config.embedding_model == "sentence-transformers/test-model"
    assert bot._loop.knowledge_config.doc_limit == 6


@pytest.mark.asyncio
async def test_agent_loop_registers_kb_search_when_knowledge_enabled(tmp_path: Path) -> None:
    fake_service = MagicMock()
    fake_service.search = AsyncMock(return_value={
        "query": "linux",
        "sufficient": True,
        "candidate_pages": [],
        "evidence_chunks": [],
    })

    with patch("nanobot.agent.loop.WebKnowledgeService", return_value=fake_service):
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_make_provider(),
            workspace=tmp_path,
            knowledge_config=KnowledgeConfig(enabled=True),
        )

    assert loop.context.knowledge_enabled is True
    assert loop.tools.has("kb_search")
    assert len(loop._system_hooks) == 1

    result = await loop.tools.execute("kb_search", {"query": "linux"})

    fake_service.search.assert_awaited_once()
    assert '"sufficient": true' in result
