"""Phase 6 memory replay runner and ablation reporting."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator, MemoryStore
from nanobot.agent.memory_db import parse_memory_markdown
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.session.manager import Session, SessionManager

REPORT_VERSION = 1

_FIXTURE_PATHS = {
    "phase0_boundary": Path("nanobot/tests/phase_0/fixtures/memory/phase0_boundary.json"),
    "v2_payload": Path("nanobot/tests/phase_3/fixtures/memory/v2_payload.json"),
}

"""Embedded fallback fixtures for the offline replay runner."""
_DEFAULT_FIXTURES: dict[str, dict[str, Any]] = {
    "phase0_boundary": {
        "name": "phase0_legacy_boundary",
        "memory_mode": "legacy",
        "session_key": "cli:fixture",
        "tokens_to_remove": 400,
        "messages": [
            {"role": "user", "content": "u1", "timestamp": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": "a1", "timestamp": "2026-01-01T00:00:01"},
            {"role": "user", "content": "u2", "timestamp": "2026-01-01T00:00:02"},
            {"role": "assistant", "content": "a2", "timestamp": "2026-01-01T00:00:03"},
            {"role": "user", "content": "u3", "timestamp": "2026-01-01T00:00:04"},
        ],
        "token_map": {"u1": 120, "a1": 120, "u2": 120, "a2": 120, "u3": 120},
        "expected": {
            "boundary_index": 4,
            "removed_tokens": 480,
            "archived_contents": ["u1", "a1", "u2", "a2"],
        },
    },
    "v2_payload": {
        "messages": [
            {
                "role": "user",
                "content": "I work on Linux, prefer concise answers, and the active project is nanobot.",
                "timestamp": "2026-04-01T11:00:00",
            },
            {
                "role": "assistant",
                "content": "Noted.",
                "timestamp": "2026-04-01T11:00:01",
            },
        ],
        "tool_arguments": {
            "history_entry": "[2026-04-01 11:00] User works on Linux, prefers concise answers, and the active project is nanobot.",
            "canonical_memories": [
                {
                    "main_class": "personal_profile",
                    "sub_class": "environment",
                    "text": "User works on Linux.",
                    "status": "active",
                },
                {
                    "main_class": "preferences",
                    "sub_class": "reply_style",
                    "text": "User prefers concise answers.",
                    "status": "active",
                },
                {
                    "main_class": "projects",
                    "sub_class": "active_project",
                    "text": "The active project is nanobot.",
                    "status": "active",
                },
            ],
        },
        "expected_memory_markdown": "# Long-term Memory\n\nThis file stores important information that should persist across sessions.\n\n## Personal Profile\n\n- environment: User works on Linux.\n\n## Preferences\n\n- reply_style: User prefers concise answers.\n\n## Constraints\n\n(Rules, boundaries, and requirements that must be respected)\n\n## Projects\n\n- active_project: The active project is nanobot.\n\n## Daily Life\n\n(Daily routines, interests, and hobbies)\n\n## Plans and Commitments\n\n(Future plans, commitments, deadlines, and to-dos)\n\n---\n\n*This file is automatically updated by nanobot when important information should be remembered.*",
        "expected_history_markdown": "[2026-04-01 11:00] User works on Linux, prefers concise answers, and the active project is nanobot.\n\n",
        "expected": {
            "main_classes": ["personal_profile", "preferences", "projects"],
            "sub_classes": ["environment", "reply_style", "active_project"],
            "texts": [
                "User works on Linux.",
                "User prefers concise answers.",
                "The active project is nanobot.",
            ],
        },
    },
}

"""不同场景虽然检查内容不同，但最终都要用同一种结构汇总进总报告。所以定义了一个统一容器"""
@dataclass
class ReplayResult:
    """One replay scenario result."""

    mode: str # 属于哪个模式
    scenario: str # 哪种回放场景，e.g., "boundary_replay", "legacy_consolidation_replay", "v2_persistence_replay", "v2_retrieval_replay", "vec_retrieval_replay", "hybrid_retrieval_replay"
    status: str # e.g., "passed", "failed", "not_implemented"
    checks: dict[str, bool] = field(default_factory=dict) # 具体检查项的结果，key是检查项名称，value是是否通过
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ScriptedProvider(LLMProvider): # 假的llm provider
    """Deterministic provider used by the replay runner."""

    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self._responses = list(responses)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_phase_fixture(name: str, fixtures_root: Path | None = None) -> dict[str, Any]:
    """Load a phase fixture from the repository, falling back to embedded defaults."""
    if name not in _FIXTURE_PATHS:
        raise KeyError(f"Unknown fixture: {name}")

    root = fixtures_root.resolve() if fixtures_root else _repo_root()
    path = root / _FIXTURE_PATHS[name]
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(json.dumps(_DEFAULT_FIXTURES[name]))


def _tool_response(name: str, tool_id: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=tool_id, name=name, arguments=arguments)],
    )


def _status_for(checks: dict[str, bool]) -> str:
    return "passed" if checks and all(checks.values()) else "failed"


def _run_boundary_replay(mode: str, fixture: dict[str, Any]) -> ReplayResult:
    with TemporaryDirectory(prefix=f"nanobot-memory-boundary-{mode}-") as tmp:
        workspace = Path(tmp)
        session = Session(key=fixture["session_key"], messages=fixture["messages"])
        consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=_ScriptedProvider([]),
            model="test-model",
            sessions=SessionManager(workspace),
            context_window_tokens=4096,
            build_messages=lambda **_kwargs: [],
            get_tool_definitions=lambda: [],
            mode=mode,
        )

        token_map = fixture["token_map"]
        with patch(
            "nanobot.agent.memory.estimate_message_tokens",
            lambda message: token_map[message["content"]],
        ):
            boundary = consolidator.pick_consolidation_boundary(
                session,
                fixture["tokens_to_remove"],
            )

    expected_boundary = (
        fixture["expected"]["boundary_index"],
        fixture["expected"]["removed_tokens"],
    )
    archived = session.messages[session.last_consolidated:fixture["expected"]["boundary_index"]]
    archived_contents = [message["content"] for message in archived]
    checks = {
        "boundary_matches": boundary == expected_boundary,
        "archived_contents_match": archived_contents == fixture["expected"]["archived_contents"],
    }
    return ReplayResult(
        mode=mode,
        scenario="boundary_replay",
        status=_status_for(checks),
        checks=checks,
        details={
            "boundary": list(boundary) if boundary else None,
            "expected_boundary": list(expected_boundary),
            "archived_contents": archived_contents,
        },
    )


async def _run_legacy_replay(fixture: dict[str, Any]) -> ReplayResult:
    with TemporaryDirectory(prefix="nanobot-memory-legacy-") as tmp:
        workspace = Path(tmp)
        store = MemoryStore(workspace, mode="legacy")
        tool_arguments = fixture["tool_arguments"]
        if "memory_update" not in tool_arguments:
            tool_arguments = {
                "history_entry": tool_arguments["history_entry"],
                "memory_update": fixture["expected_memory_markdown"],
            }
        provider = _ScriptedProvider([
            _tool_response("save_memory", "legacy_call_1", tool_arguments),
        ])

        result = await store.consolidate(fixture["messages"], provider, "test-model")
        memory_text = store.memory_file.read_text(encoding="utf-8")
        history_text = store.history_file.read_text(encoding="utf-8")
        checks = {
            "consolidate_returned_true": result is True,
            "memory_view_matches": memory_text == fixture["expected_memory_markdown"],
            "history_view_matches": history_text == fixture["expected_history_markdown"],
        }
        return ReplayResult(
            mode="legacy",
            scenario="legacy_consolidation_replay",
            status=_status_for(checks),
            checks=checks,
            details={
                "message_count": len(fixture["messages"]),
                "memory_bytes": len(memory_text.encode("utf-8")),
                "history_bytes": len(history_text.encode("utf-8")),
            },
        )


async def _run_v2_replay(fixture: dict[str, Any]) -> ReplayResult:
    with TemporaryDirectory(prefix="nanobot-memory-v2-") as tmp:
        workspace = Path(tmp)
        store = MemoryStore(workspace, mode="v2")
        provider = _ScriptedProvider([
            _tool_response("save_memory_structured", "v2_call_1", fixture["tool_arguments"]),
        ])

        result = await store.consolidate(fixture["messages"], provider, "test-model")
        assert store.v2_db is not None
        memory_text = store.memory_file.read_text(encoding="utf-8")
        history_text = store.history_file.read_text(encoding="utf-8")
        canonical = store.v2_db.list_canonical_memories()
        checks = {
            "consolidate_returned_true": result is True,
            "db_exists": store.v2_db.db_path.exists(),
            "memory_view_matches": memory_text == fixture["expected_memory_markdown"],
            "history_view_matches": history_text == fixture["expected_history_markdown"],
            "canonical_texts_match": [item["text"] for item in canonical] == fixture["expected"]["texts"],
        }
        return ReplayResult(
            mode="v2",
            scenario="v2_persistence_replay",
            status=_status_for(checks),
            checks=checks,
            details={
                "message_count": len(fixture["messages"]),
                "canonical_count": len(canonical),
            },
        )


async def _run_v2_retrieval_replay(fixture: dict[str, Any]) -> ReplayResult:
    with TemporaryDirectory(prefix="nanobot-memory-v2-retrieval-") as tmp:
        workspace = Path(tmp)
        store = MemoryStore(workspace, mode="v2")
        provider = _ScriptedProvider([
            _tool_response("save_memory_structured", "v2_retrieval_seed_call_1", fixture["tool_arguments"]),
        ])

        await store.consolidate(fixture["messages"], provider, "test-model")
        builder = ContextBuilder(workspace, memory_mode="v2")
        query = "Please answer concisely."
        messages = builder.build_messages(history=[], current_message=query)
        system_prompt = messages[0]["content"]
        checks = {
            "core_memory_header_present": "## Core Memory" in system_prompt,
            "profile_memory_present": "[personal_profile/environment] User works on Linux." in system_prompt,
            "relevant_memory_present": "[preferences/reply_style] User prefers concise answers." in system_prompt,
            "retrieved_memory_header_absent_for_core_only_match": "## Retrieved Memory" not in system_prompt,
            "unrelated_project_not_injected": "The active project is nanobot." not in system_prompt,
        }
        return ReplayResult(
            mode="v2",
            scenario="v2_retrieval_replay",
            status=_status_for(checks),
            checks=checks,
            details={
                "query": query,
                "prompt_contains_memory": "# Memory" in system_prompt,
                "prompt_strategy": "core_plus_fts",
            },
        )


def _not_implemented_result(mode: str) -> ReplayResult:
    return ReplayResult(
        mode=mode,
        scenario=f"{mode}_retrieval_replay",
        status="not_implemented",
        checks={},
        details={
            "reason": (
                f"{mode} retrieval belongs to Phase 5 sqlite-vec work and is not implemented "
                "in the current codebase."
            ),
        },
    )


def _summarize_by_status(results: list[ReplayResult]) -> dict[str, int]:
    summary = {"passed": 0, "failed": 0, "not_implemented": 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    summary["total"] = len(results)
    return summary


def _build_mode_summary(results: list[ReplayResult]) -> dict[str, dict[str, Any]]:
    mode_summary: dict[str, dict[str, Any]] = {}
    for result in results:
        entry = mode_summary.setdefault(
            result.mode,
            {"passed": 0, "failed": 0, "not_implemented": 0, "status": "passed"},
        )
        entry[result.status] = entry.get(result.status, 0) + 1

    for mode, entry in mode_summary.items():
        if entry.get("failed", 0):
            entry["status"] = "failed"
        elif entry.get("passed", 0):
            entry["status"] = "passed"
        else:
            entry["status"] = "not_implemented"
        entry["mode"] = mode
    return mode_summary


async def _run_phase6_ablation_async(fixtures_root: Path | None = None) -> dict[str, Any]:
    phase0 = load_phase_fixture("phase0_boundary", fixtures_root)
    v2 = load_phase_fixture("v2_payload", fixtures_root)

    results = [
        _run_boundary_replay("legacy", phase0),
        _run_boundary_replay("v2", phase0),
        await _run_legacy_replay(v2),
        await _run_v2_replay(v2),
        await _run_v2_retrieval_replay(v2),
        _not_implemented_result("vec"),
        _not_implemented_result("hybrid"),
    ]

    return {
        "report_version": REPORT_VERSION,
        "fixtures_root": str(fixtures_root.resolve() if fixtures_root else _repo_root()),
        "summary": _summarize_by_status(results),
        "mode_summary": _build_mode_summary(results),
        "results": [result.to_dict() for result in results],
    }


def run_phase6_ablation(fixtures_root: Path | None = None) -> dict[str, Any]:
    """Run the offline Phase 6 replay suite and return a fixed-format report."""
    return asyncio.run(_run_phase6_ablation_async(fixtures_root))


def render_phase6_report_markdown(report: dict[str, Any]) -> str:
    """Render the fixed-format report as Markdown."""
    summary = report["summary"]
    lines = [
        "# Memory Phase 6 Ablation Report",
        "",
        f"- Report Version: {report['report_version']}",
        f"- Fixtures Root: `{report['fixtures_root']}`",
        f"- Total Scenarios: {summary['total']}",
        f"- Passed: {summary.get('passed', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Not Implemented: {summary.get('not_implemented', 0)}",
        "",
        "## Mode Summary",
        "",
        "| Mode | Status | Passed | Failed | Not Implemented |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for mode in ("legacy", "v2", "vec", "hybrid"):
        item = report["mode_summary"].get(mode, {})
        lines.append(
            f"| {mode} | {item.get('status', 'n/a')} | {item.get('passed', 0)} | "
            f"{item.get('failed', 0)} | {item.get('not_implemented', 0)} |"
        )

    lines.extend([
        "",
        "## Scenario Results",
        "",
        "| Mode | Scenario | Status | Checks |",
        "| --- | --- | --- | --- |",
    ])
    for result in report["results"]:
        check_count = len(result["checks"])
        check_passed = sum(1 for ok in result["checks"].values() if ok)
        lines.append(
            f"| {result['mode']} | {result['scenario']} | {result['status']} | "
            f"{check_passed}/{check_count} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- `vec` and `hybrid` are intentionally marked `not_implemented` until Phase 5 lands.",
        "- This report is fixture-driven and does not call a live provider.",
    ])
    return "\n".join(lines)


def save_phase6_report(report: dict[str, Any], save_dir: Path) -> dict[str, Path]:
    """Save both JSON and Markdown copies of the Phase 6 report."""
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "memory_phase6_ablation_report.json"
    markdown_path = save_dir / "memory_phase6_ablation_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_phase6_report_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


# ============================================================================
# Memory Extraction Evaluation
# ============================================================================

MemoryProviderFactory = Callable[[dict[str, Any]], LLMProvider]


