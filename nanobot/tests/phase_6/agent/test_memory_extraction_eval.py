# Phase 6 test content:
# - verifies the v2-only extraction evaluation runs end-to-end over temporary workspaces
# - verifies prior_memory is injected into the v2 consolidation prompt
# - verifies extraction metrics, markdown rendering, and report saving are deterministic
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_6/agent/test_memory_extraction_eval.py

from __future__ import annotations

import json
from typing import Any

import pytest

from nanobot.agent.memory_eval import (
    load_memory_extraction_cases,
    render_memory_extraction_report_markdown,
    run_memory_extraction_eval,
    save_memory_extraction_report,
)
from nanobot.agent.memory_sync import MarkdownValidationError, render_memory_markdown
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _ReplayProvider(LLMProvider):
    def __init__(self, response: LLMResponse, request_log: list[list[dict[str, Any]]] | None = None):
        super().__init__()
        self._response = response
        self._request_log = request_log

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
        if self._request_log is not None:
            self._request_log.append(messages)
        return self._response

    def get_default_model(self) -> str:
        return "test-model"


def _memory(*items: tuple[str, str, str]) -> list[dict[str, Any]]:
    return [
        {
            "main_class": main_class,
            "sub_class": sub_class,
            "text": text,
        }
        for main_class, sub_class, text in items
    ]


def _markdown(snapshot: list[dict[str, Any]]) -> str:
    return render_memory_markdown(snapshot)


def _conversation(*contents: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index, content in enumerate(contents):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append({
            "role": role,
            "content": content,
            "timestamp": f"2026-04-02T10:00:0{index}",
        })
    return messages


def _build_cases() -> list[dict[str, Any]]:
    old_pref = _memory(("preferences", "reply_style", "User prefers concise answers."))
    new_pref = _memory(("preferences", "reply_style", "User prefers concise bullet-point answers."))
    profile_and_constraint = _memory(
        ("personal_profile", "environment", "User works on Linux."),
        ("constraints", "tooling", "Avoid interactive git commands."),
    )
    profile_only = _memory(("personal_profile", "environment", "User works on Linux."))
    project_pair = _memory(
        ("projects", "active_project", "The active project is nanobot."),
        ("projects", "active_project", "The active project is memory-v2-eval."),
    )

    return [
        {
            "case_id": "add_new_memory",
            "case_type": "add_new_memory",
            "prior_memory_markdown": "",
            "conversation": _conversation("Remember that I work on Linux.", "Noted."),
            "gold_snapshot": profile_only,
            "mock_responses": {
                "v2": {
                    "history_entry": "[2026-04-02 10:00] User works on Linux.",
                    "canonical_memories": profile_only,
                },
            },
        },
        {
            "case_id": "update_existing_memory",
            "case_type": "update_existing_memory",
            "prior_memory_markdown": _markdown(old_pref),
            "conversation": _conversation("Update my preference to concise bullet-point answers.", "Understood."),
            "gold_snapshot": new_pref,
            "mock_responses": {
                "v2": {
                    "history_entry": "[2026-04-02 10:02] User now prefers concise bullet-point answers.",
                    "canonical_memories": new_pref,
                },
            },
        },
        {
            "case_id": "delete_outdated_memory",
            "case_type": "delete_outdated_memory",
            "prior_memory_markdown": _markdown(profile_and_constraint),
            "conversation": _conversation("The Linux environment still matters, but the git restriction is no longer needed.", "Got it."),
            "gold_snapshot": profile_only,
            "mock_responses": {
                "v2": {
                    "history_entry": "[2026-04-02 10:04] The git restriction is outdated.",
                    "canonical_memories": profile_only,
                },
            },
        },
        {
            "case_id": "same_subclass_multiple_memories",
            "case_type": "same_subclass_multiple_memories",
            "prior_memory_markdown": "",
            "conversation": _conversation("Track nanobot and memory-v2-eval as active projects.", "Noted."),
            "gold_snapshot": project_pair,
            "mock_responses": {
                "v2": {
                    "history_entry": "[2026-04-02 10:06] User has two active projects.",
                    "canonical_memories": project_pair,
                },
            },
        },
        {
            "case_id": "no_op_case",
            "case_type": "no_op_case",
            "prior_memory_markdown": _markdown(profile_only),
            "conversation": _conversation("Thanks for the help today.", "Happy to help."),
            "gold_snapshot": profile_only,
            "mock_responses": {
                "v2": {
                    "history_entry": "[2026-04-02 10:08] Routine thanks exchange.",
                    "canonical_memories": profile_only,
                },
            },
        },
    ]


def _tool_response(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=f"{name}_call", name=name, arguments=arguments)],
    )


def _v2_factory(case: dict[str, Any], request_log: list[list[dict[str, Any]]] | None = None) -> _ReplayProvider:
    return _ReplayProvider(
        _tool_response("save_memory_structured", case["mock_responses"]["v2"]),
        request_log=request_log,
    )


def test_run_memory_extraction_eval_scores_v2_snapshot_only() -> None:
    cases = _build_cases()
    report = run_memory_extraction_eval(
        cases=cases,
        v2_provider_factory=lambda case: _v2_factory(case),
        model="test-model",
    )

    metrics = report["metrics"]

    assert report["evaluation"] == "memory_extraction"
    assert report["temporary_workspace_execution"] is True
    assert report["mode"] == "v2"
    assert report["total_cases"] == 5

    assert metrics["snapshot_precision"] == pytest.approx(1.0)
    assert metrics["snapshot_recall"] == pytest.approx(1.0)
    assert metrics["snapshot_f1"] == pytest.approx(1.0)
    assert metrics["exact_match_rate"] == pytest.approx(1.0)
    assert metrics["addition_recall"] == pytest.approx(1.0)
    assert metrics["deletion_accuracy"] == pytest.approx(1.0)
    assert metrics["stale_retention_rate"] == pytest.approx(0.0)
    assert metrics["multi_memory_preservation_rate"] == pytest.approx(1.0)
    assert report["cases"][2]["v2"]["metrics"]["exact_match"] is True


@pytest.mark.parametrize("formatting", ["canonical", "trailing_newline", "extra_blank_lines"])
def test_run_memory_extraction_eval_loads_cases_path_and_injects_prior_memory(
    tmp_path, formatting
) -> None:
    cases = _build_cases()[1:2]
    if formatting == "trailing_newline":
        cases[0]["prior_memory_markdown"] += "\n"
    elif formatting == "extra_blank_lines":
        cases[0]["prior_memory_markdown"] = cases[0]["prior_memory_markdown"].replace(
            "\n\n", "\n\n\n"
        )
    cases_path = tmp_path / "memory_extraction_cases.json"
    cases_path.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_memory_extraction_cases(cases_path)
    assert loaded[0]["case_id"] == "update_existing_memory"

    v2_requests: list[list[dict[str, Any]]] = []
    report = run_memory_extraction_eval(
        cases_path=cases_path,
        v2_provider_factory=lambda case: _v2_factory(case, request_log=v2_requests),
        model="test-model",
    )

    assert report["cases_source"] == str(cases_path.resolve())
    assert len(v2_requests) == 1
    v2_prompt = v2_requests[0][1]["content"]
    assert "User prefers concise answers." in v2_prompt
    assert report["metrics"]["exact_match_rate"] == pytest.approx(1.0)


def test_run_memory_extraction_eval_rejects_legacy_prior_markdown_without_subclass(tmp_path) -> None:
    cases = _build_cases()[1:2]
    cases[0]["prior_memory_markdown"] = (
        "# Long-term Memory\n\n"
        "## Preferences\n\n"
        "- User prefers concise answers.\n"
    )

    with pytest.raises(MarkdownValidationError, match="canonical order"):
        run_memory_extraction_eval(
            cases=cases,
            v2_provider_factory=lambda case: _v2_factory(case),
            model="test-model",
        )


def test_render_and_save_memory_extraction_report(tmp_path) -> None:
    report = run_memory_extraction_eval(
        cases=_build_cases(),
        v2_provider_factory=lambda case: _v2_factory(case),
        model="test-model",
    )

    rendered = render_memory_extraction_report_markdown(report)
    saved = save_memory_extraction_report(report, tmp_path)

    assert "# Memory Extraction Evaluation Report" in rendered
    assert "## V2 Metrics" in rendered
    assert "same_subclass_multiple_memories" in rendered
    assert saved["json"].exists()
    assert saved["markdown"].exists()
    assert json.loads(saved["json"].read_text(encoding="utf-8"))["total_cases"] == 5
    assert "# Memory Extraction Evaluation Report" in saved["markdown"].read_text(encoding="utf-8")
