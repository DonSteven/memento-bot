import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.context_budget import ContextBudgetError, fit_request, request_tokens
from nanobot.agent.hook import AgentHook
from nanobot.agent.memory_db import MemoryContext, MemoryRecord
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.tests.memory_test_utils import TestAgentLoop


def provider():
    return SimpleNamespace(generation=GenerationSettings(max_tokens=100),
                           get_default_model=lambda: "test",
                           chat_with_retry=AsyncMock(return_value=LLMResponse(content="ok")))


@pytest.mark.asyncio
async def test_required_content_over_limit_stops_model_even_with_hook():
    p = provider()
    class AddContent(AgentHook):
        async def before_iteration(self, context):
            context.messages.append({"role": "user", "content": "very long " * 2000})
    result = await AgentRunner(p).run(AgentRunSpec(
        initial_messages=[{"role": "system", "content": "Core unchanged"}], tools=ToolRegistry(),
        model="test", max_iterations=2, hook=AddContent(), context_window_tokens=200))
    assert result.stop_reason == "context_limit"
    assert "Core unchanged" == result.messages[0]["content"]
    p.chat_with_retry.assert_not_awaited()


def test_fit_removes_whole_dynamic_records_and_preserves_core():
    p = provider()
    core = MemoryRecord.create("constraints", "rule", "Always retain this")
    dynamic = MemoryRecord.create("projects", "work", "long content " * 1000)
    memory = MemoryContext((core,), (dynamic,))
    messages = [{"role": "system", "content": memory.render_block()}, {"role": "user", "content": "hello"}]
    result = fit_request(p, "test", messages, [], 300, 100, memory)
    assert result == MemoryContext((core,), ())
    assert messages[0]["content"] == result.render_block()


@pytest.mark.asyncio
async def test_tool_result_is_checked_before_next_model_call():
    p = provider()
    p.chat_with_retry.return_value = LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="1", name="large", arguments={})])
    tools = ToolRegistry()
    tools.execute = AsyncMock(return_value="large response " * 2000)
    result = await AgentRunner(p).run(AgentRunSpec(initial_messages=[{"role": "user", "content": "hi"}],
        tools=tools, model="test", max_iterations=3, context_window_tokens=300))
    assert result.stop_reason == "context_limit"
    assert p.chat_with_retry.await_count == 1


def test_evidence_budget_preserves_parent_child_links_and_invalidates_assessment():
    p = provider()
    evidence = {"status": "sufficient", "sufficient": True, "missing_points": [], "reason": "covered",
                "parents": [{"parent_id": 1, "url": "https://example.com", "text": "short"},
                            {"parent_id": 2, "url": "https://example.org", "text": "huge " * 2000}],
                "children": [{"parent_id": 1, "text": "short"}, {"parent_id": 2, "text": "huge " * 1000}]}
    messages = [{"role": "tool", "name": "kb_search", "content": json.dumps(evidence)}]
    fit_request(p, "test", messages, [], 500, 100)
    reduced = json.loads(messages[0]["content"])
    assert reduced["children"] == evidence["children"][:1]
    assert reduced["parents"] == evidence["parents"][:1]
    assert not reduced["sufficient"] and reduced["budget_limited"]


@pytest.mark.asyncio
async def test_core_overflow_prevents_history_extraction_and_returns_visible_error(tmp_path):
    p = provider()
    loop = TestAgentLoop(MessageBus(), p, tmp_path, context_window_tokens=200)
    loop.memory_service.prepare_context = AsyncMock(return_value=MemoryContext((
        MemoryRecord.create("constraints", "large", "core " * 1000),), ()))
    loop.memory_consolidator.maybe_consolidate_by_tokens = AsyncMock()
    response = await loop.process_direct("hello")
    assert response.metadata["stop_reason"] == "context_limit"
    loop.memory_consolidator.maybe_consolidate_by_tokens.assert_not_awaited()
    p.chat_with_retry.assert_not_awaited()


def test_output_reservation_and_tools_are_counted():
    p = provider()
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"description": "large " * 100}]
    needed = request_tokens(p, "test", messages, tools)
    fit_request(p, "test", deepcopy(messages), tools, needed + 100, 100)
    with pytest.raises(ContextBudgetError):
        fit_request(p, "test", messages, tools, needed + 99, 100)


def test_literal_special_tokens_can_be_counted():
    p = provider()
    messages = [{"role": "user", "content": "Explain <|endoftext|> literally"}]
    assert request_tokens(p, "test", messages, []) > 0


def test_multimodal_payload_is_not_omitted_from_local_estimation():
    p = provider()
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "abcdef" * 1000}}]}]
    assert request_tokens(p, "test", messages, []) > 1000
