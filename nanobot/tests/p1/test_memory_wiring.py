from __future__ import annotations

import inspect
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nanobot.agent.memory_db import MemoryWriteResult
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config
from nanobot.config.schema import MemoryConfig
from nanobot.providers.base import GenerationSettings
from nanobot.tests.memory_test_utils import TestAgentLoop as AgentLoop


def _provider():
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    return provider


def test_agent_loop_uses_one_memory_service_for_context_and_archival(tmp_path) -> None:
    loop = AgentLoop(MessageBus(), _provider(), tmp_path)
    assert loop.memory_consolidator.memory_service is loop.memory_service
    assert "memory_mode" not in inspect.signature(AgentLoop).parameters


def test_memory_mode_configuration_was_removed() -> None:
    assert MemoryConfig().embedding.model == "text-embedding-v4"
    assert MemoryConfig().dynamic_top_k == 5
    with pytest.raises(ValidationError):
        MemoryConfig.model_validate({"mode": "v2"})


def test_config_loader_reports_removed_memory_mode_instead_of_using_defaults(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"memory": {"mode": "legacy"}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="memory.mode setting was removed"):
        load_config(config_path)


@pytest.mark.asyncio
async def test_new_keeps_pending_session_when_database_archive_fails(tmp_path) -> None:
    loop = AgentLoop(MessageBus(), _provider(), tmp_path)
    session = loop.sessions.get_or_create("cli:test")
    session.add_message("user", "remember this")
    loop.sessions.save(session)
    loop.memory_consolidator.archive_messages = AsyncMock(
        return_value=MemoryWriteResult(False, False, 0, "database unavailable")
    )

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/new")
    )

    assert response is not None and "archival failed" in response.content
    assert [item["content"] for item in loop.sessions.get_or_create("cli:test").messages] == [
        "remember this"
    ]


@pytest.mark.asyncio
async def test_new_clears_committed_session_when_only_view_export_fails(tmp_path) -> None:
    loop = AgentLoop(MessageBus(), _provider(), tmp_path)
    session = loop.sessions.get_or_create("cli:test")
    session.add_message("user", "remember this")
    loop.sessions.save(session)
    loop.memory_consolidator.archive_messages = AsyncMock(
        return_value=MemoryWriteResult(True, False, 1, "disk full")
    )

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="test", content="/new")
    )

    assert response is not None and "memory was saved" in response.content
    assert loop.sessions.get_or_create("cli:test").messages == []
