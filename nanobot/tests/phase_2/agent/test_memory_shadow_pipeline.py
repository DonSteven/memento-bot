# Phase 2 test content:
# - verifies the shadow pipeline writes structured sidecar artifacts into memory/shadow
# - verifies canonical memories, evidence links, rendered views, and debug payload are all emitted
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_2/agent/test_memory_shadow_pipeline.py

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from nanobot.agent.memory_pipeline import ShadowMemoryPipeline
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "memory" / f"{name}.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_shadow_pipeline_writes_sidecar_artifacts(tmp_path: Path) -> None:
    fixture = _load_fixture("shadow_payload")
    provider = AsyncMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="shadow_call_1",
                    name="save_memory_shadow",
                    arguments=fixture["tool_arguments"],
                )
            ],
        )
    )

    pipeline = ShadowMemoryPipeline(tmp_path)

    result = __import__("asyncio").run(
        pipeline.ingest_chunk(fixture["messages"], provider, "test-model")
    )

    assert result is True
    assert pipeline.db.db_path.exists()
    assert pipeline.db.memory_file.read_text(encoding="utf-8") == fixture["expected_memory_markdown"]
    assert pipeline.db.history_file.read_text(encoding="utf-8") == fixture["expected_history_markdown"]

    raw_events = pipeline.db.list_raw_events()
    memories = pipeline.db.list_canonical_memories()
    evidence = pipeline.db.list_evidence_links()

    assert len(raw_events) == 1
    assert len(memories) == 2
    assert len(evidence) == 2
    assert json.loads(pipeline.debug_file.read_text(encoding="utf-8"))["history_entry"] == fixture["tool_arguments"]["history_entry"]
