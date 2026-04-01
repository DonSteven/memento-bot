# Phase 2 test content:
# - verifies the shadow pipeline writes structured sidecar artifacts into memory/shadow
# - verifies canonical memories, evidence links, rendered views, and debug payload are all emitted
# - verifies repeated replay is stable after ignoring non-deterministic event metadata
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


def _normalize_raw_events(events: list[dict]) -> list[dict]:
    normalized = []
    for event in events:
        normalized.append({
            "session_key": event["session_key"],
            "history_text": event["history_text"],
            "plain_text": event["plain_text"],
            "candidate_type": event["candidate_type"],
            "extracted_json": event["extracted_json"],
        })
    return normalized


def _normalize_evidence_links(links: list[dict], memories: list[dict]) -> list[tuple[str, float]]:
    memory_ids = {link["memory_id"] for link in links}
    assert memory_ids.issubset({memory["memory_id"] for memory in memories})
    return sorted((link["memory_id"], link["weight"]) for link in links)


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


def test_shadow_pipeline_replay_is_stable_ignoring_event_metadata(tmp_path: Path) -> None:
    fixture = _load_fixture("shadow_payload")

    def _provider():
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
        return provider

    pipeline_one = ShadowMemoryPipeline(tmp_path / "run_one")
    pipeline_two = ShadowMemoryPipeline(tmp_path / "run_two")

    __import__("asyncio").run(pipeline_one.ingest_chunk(fixture["messages"], _provider(), "test-model"))
    __import__("asyncio").run(pipeline_two.ingest_chunk(fixture["messages"], _provider(), "test-model"))

    memories_one = pipeline_one.db.list_canonical_memories()
    memories_two = pipeline_two.db.list_canonical_memories()
    raw_one = pipeline_one.db.list_raw_events()
    raw_two = pipeline_two.db.list_raw_events()
    evidence_one = pipeline_one.db.list_evidence_links()
    evidence_two = pipeline_two.db.list_evidence_links()

    assert pipeline_one.db.memory_file.read_text(encoding="utf-8") == pipeline_two.db.memory_file.read_text(encoding="utf-8")
    assert pipeline_one.db.history_file.read_text(encoding="utf-8") == pipeline_two.db.history_file.read_text(encoding="utf-8")
    assert memories_one == memories_two
    assert _normalize_raw_events(raw_one) == _normalize_raw_events(raw_two)
    assert _normalize_evidence_links(evidence_one, memories_one) == _normalize_evidence_links(evidence_two, memories_two)
