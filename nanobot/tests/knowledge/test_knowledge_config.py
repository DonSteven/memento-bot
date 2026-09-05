# Knowledge test content:
# - verifies external web knowledge config defaults to disabled
# - verifies Nanobot.from_config preserves the configured knowledge settings
# - verifies AgentLoop wires kb_search and system hooks when knowledge is enabled
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/knowledge/test_knowledge_config.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.hook import AgentHook
from nanobot.agent.knowledge import WebKnowledgeHook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.knowledge import KnowledgeSearchTool
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
            "embedding": {
                "provider": "dashscope",
                "model": "text-embedding-v4",
                "apiBase": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "dimensions": 1024,
            },
            "docLimit": 6,
        },
    }))
    return config_path


def test_config_defaults_to_disabled_web_knowledge() -> None:
    config = Config()
    assert config.knowledge.enabled is False
    assert config.knowledge.chunk_chars == 800
    assert config.knowledge.chunk_overlap_chars == 150
    assert config.knowledge.doc_limit == 10
    assert config.knowledge.evidence_limit == 5
    assert config.knowledge.embedding.model == "text-embedding-v4"
    assert config.knowledge.embedding.dimensions == 1024
    assert config.knowledge.rerank.model == "qwen3-rerank"
    assert config.knowledge.rerank.enabled is True
    assert config.knowledge.child_fts_limit == 24
    assert config.knowledge.child_vec_limit == 24
    assert config.knowledge.rerank_child_pool == 24
    assert config.knowledge.max_children_per_parent == 2


def test_nanobot_from_config_passes_knowledge_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    assert bot._loop.knowledge_config.enabled is False
    assert bot._loop.knowledge_config.embedding.model == "text-embedding-v4"
    assert bot._loop.knowledge_config.doc_limit == 6


def test_kb_search_tool_defaults_match_production_limits() -> None:
    properties = KnowledgeSearchTool.parameters["properties"]

    assert properties["docLimit"]["default"] == 10
    assert properties["docLimit"]["maximum"] == 100
    assert properties["evidenceLimit"]["default"] == 5
    assert properties["evidenceLimit"]["maximum"] == 100


@pytest.mark.asyncio
async def test_agent_loop_registers_kb_search_when_knowledge_enabled(tmp_path: Path) -> None:
    fake_service = MagicMock()
    fake_service.search = AsyncMock(return_value={
        "query": "linux",
        "sufficient": True,
        "candidate_parents": [],
        "evidence_chunks": [],
    })

    with patch("nanobot.agent.loop.WebKnowledgeService", return_value=fake_service) as service_cls:
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_make_provider(),
            workspace=tmp_path,
            knowledge_config=KnowledgeConfig(enabled=True),
            knowledge_api_key="sk-dashscope-test",
        )

    assert loop.context.knowledge_enabled is True
    assert loop.tools.has("kb_search")
    assert len(loop._system_hooks) == 1
    assert service_cls.call_args.kwargs["api_key"] == "sk-dashscope-test"

    result = await loop.tools.execute("kb_search", {"query": "linux"})

    fake_service.search.assert_awaited_once()
    payload = json.loads(result)
    assert payload == {
        "query": "linux",
        "sufficient": True,
        "candidate_parents": [],
        "evidence_chunks": [],
    }


@pytest.mark.asyncio
async def test_nanobot_run_keeps_system_hooks_ahead_of_sdk_hooks(tmp_path: Path) -> None:
    fake_service = MagicMock()
    user_hook = AgentHook()

    with patch("nanobot.agent.loop.WebKnowledgeService", return_value=fake_service):
        loop = AgentLoop(
            bus=MessageBus(),
            provider=_make_provider(),
            workspace=tmp_path,
            knowledge_config=KnowledgeConfig(enabled=True),
        )

    async def _fake_run(spec):
        extras = spec.hook._extras._hooks
        assert isinstance(extras[0], WebKnowledgeHook)
        assert extras[1] is user_hook
        return AgentRunResult(final_content="ok", messages=list(spec.initial_messages))

    loop.runner.run = AsyncMock(side_effect=_fake_run)

    bot = Nanobot(loop)
    result = await bot.run("hello", hooks=[user_hook])

    assert result.content == "ok"
    assert loop._extra_hooks == []
