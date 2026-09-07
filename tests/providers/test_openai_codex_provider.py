"""Request format coverage for the Codex Responses transport (no external calls)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.responses.tool_choice_function import ToolChoiceFunction

from nanobot.providers import openai_codex_provider as codex


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", "none", "named"])
async def test_codex_converts_chat_tool_choice_to_responses(monkeypatch, stream, choice):
    monkeypatch.setattr(
        codex.asyncio, "to_thread",
        AsyncMock(return_value=SimpleNamespace(account_id="test", access="test")),
    )
    request = AsyncMock(return_value=("", [], "stop"))
    monkeypatch.setattr(codex, "_request_codex", request)
    provider = codex.OpenAICodexProvider()
    chat = provider.chat_stream if stream else provider.chat
    tool_choice = (
        {"type": "function", "function": {"name": "report_evidence_assessment"}}
        if choice == "named" else choice
    )

    response = await chat(
        messages=[{"role": "user", "content": "Assess the evidence"}],
        tools=[{"type": "function", "function": {
            "name": "report_evidence_assessment", "parameters": {"type": "object"},
        }}],
        tool_choice=tool_choice,
    )

    assert response.finish_reason == "stop"
    request.assert_awaited_once()
    body = request.call_args.args[2]
    if choice == "named":
        validated = ToolChoiceFunction.model_validate(body["tool_choice"])
        assert validated.name == "report_evidence_assessment"
        assert "function" not in body["tool_choice"]
    else:
        assert body["tool_choice"] == choice
    assert body["tools"][0]["name"] == "report_evidence_assessment"
