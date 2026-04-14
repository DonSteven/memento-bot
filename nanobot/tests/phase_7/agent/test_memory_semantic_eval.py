# Phase 7 test content:
# - verifies Phase 7 semantic fixtures load and filter correctly
# - verifies bucketed Hungarian matching respects main_class boundaries and thresholds
# - verifies the mocked-provider v2 runner produces deterministic semantic metrics and reports
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_7/agent/test_memory_semantic_eval.py

from __future__ import annotations

import asyncio
import itertools
import json
from typing import Any

import pytest

from nanobot.agent import memory_semantic_eval as semantic_eval
from nanobot.agent.memory_semantic_eval import (
    load_memory_semantic_cases,
    render_memory_v2_semantic_report_markdown,
    run_memory_v2_semantic_eval,
    save_memory_v2_semantic_report,
)
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]


class _ReplayProvider(LLMProvider):
    def __init__(self, arguments: dict[str, Any]):
        super().__init__()
        self._arguments = arguments

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
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="save_memory_structured_call",
                    name="save_memory_structured",
                    arguments=self._arguments,
                )
            ],
        )

    def get_default_model(self) -> str:
        return "test-model"


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[str]):
        super().__init__()
        self._responses = list(responses)
        self.call_count = 0

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
        self.call_count += 1
        content = self._responses.pop(0) if self._responses else '{"verdict":"no_match","reason":"default"}'
        return LLMResponse(content=content)

    def get_default_model(self) -> str:
        return "test-model"


def _message(role: str, content: str, timestamp: str) -> dict[str, str]:
    return {"role": role, "content": content, "timestamp": timestamp}


def _memory(main_class: str, sub_class: str, text: str) -> dict[str, str]:
    return {
        "main_class": main_class,
        "sub_class": sub_class,
        "text": text,
    }


def _case_main_classes(case: dict[str, Any]) -> set[str]:
    classes: set[str] = set()
    for field in (
        "prior_snapshot",
        "gold_final_snapshot",
        "gold_extracted_snapshot",
        "gold_preserved_snapshot",
    ):
        for item in case.get(field, []):
            main_class = item.get("main_class")
            if isinstance(main_class, str) and main_class:
                classes.add(main_class)
    return classes


@pytest.fixture(autouse=True)
def _patch_hungarian(monkeypatch) -> None:
    def _fake_linear_sum_assignment(cost_matrix: list[list[float]]) -> tuple[list[int], list[int]]:
        row_count = len(cost_matrix)
        col_count = len(cost_matrix[0]) if row_count else 0
        if row_count == 0 or col_count == 0:
            return [], []

        pair_count = min(row_count, col_count)
        best_rows: list[int] = []
        best_cols: list[int] = []
        best_cost: float | None = None

        if row_count <= col_count:
            row_indexes = list(range(row_count))
            for col_indexes in itertools.permutations(range(col_count), pair_count):
                total_cost = sum(cost_matrix[row][col] for row, col in zip(row_indexes, col_indexes))
                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_rows = row_indexes
                    best_cols = list(col_indexes)
        else:
            col_indexes = list(range(col_count))
            for row_indexes in itertools.permutations(range(row_count), pair_count):
                total_cost = sum(cost_matrix[row][col] for row, col in zip(row_indexes, col_indexes))
                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_rows = list(row_indexes)
                    best_cols = col_indexes

        return best_rows, best_cols

    monkeypatch.setattr(semantic_eval, "_load_hungarian", lambda: _fake_linear_sum_assignment)


def test_load_memory_semantic_cases_filters_by_level_and_scenario() -> None:
    all_cases = load_memory_semantic_cases()
    l0_cases = load_memory_semantic_cases(level="L0")
    l1_cases = load_memory_semantic_cases(level="L1")
    extraction_cases = load_memory_semantic_cases(scenario="extraction")
    overwrite_cases = load_memory_semantic_cases(scenario="overwrite_preservation")

    assert len(all_cases) == 16
    assert {case["level"] for case in all_cases} == {"L0", "L1"}
    assert {case["scenario"] for case in all_cases} == {"extraction", "overwrite_preservation"}
    assert len(l0_cases) == 8
    assert {case["level"] for case in l0_cases} == {"L0"}
    assert len(l1_cases) == 8
    assert {case["level"] for case in l1_cases} == {"L1"}
    assert len(extraction_cases) == 8
    assert {case["scenario"] for case in extraction_cases} == {"extraction"}
    assert len(overwrite_cases) == 8
    assert {case["scenario"] for case in overwrite_cases} == {"overwrite_preservation"}
    assert len(load_memory_semantic_cases(level="L0", scenario="extraction")) == 4
    assert len(load_memory_semantic_cases(level="L0", scenario="overwrite_preservation")) == 4
    assert len(load_memory_semantic_cases(level="L1", scenario="extraction")) == 4
    assert len(load_memory_semantic_cases(level="L1", scenario="overwrite_preservation")) == 4


def test_phase7_fixtures_have_expected_minimum_conversation_lengths() -> None:
    l0_cases = load_memory_semantic_cases(level="L0")
    l1_cases = load_memory_semantic_cases(level="L1")

    assert all(len(case["conversation"]) >= 6 for case in l0_cases)
    assert all(len(case["conversation"]) >= 50 for case in l1_cases)
    assert all(len(case["conversation"]) % 2 == 0 for case in l1_cases)


def test_phase7_scenarios_cover_all_main_classes() -> None:
    expected = {
        "personal_profile",
        "preferences",
        "constraints",
        "projects",
        "daily_life",
        "plans_commitments",
    }

    extraction_cases = load_memory_semantic_cases(scenario="extraction")
    overwrite_cases = load_memory_semantic_cases(scenario="overwrite_preservation")

    extraction_covered = set().union(*(_case_main_classes(case) for case in extraction_cases))
    overwrite_covered = set().union(*(_case_main_classes(case) for case in overwrite_cases))

    assert extraction_covered == expected
    assert overwrite_covered == expected


def test_match_snapshots_never_crosses_main_class_boundaries() -> None:
    predicted = [_memory("personal_profile", "environment", "User works on Linux.")]
    gold = [_memory("preferences", "environment", "User works on Linux.")]
    embedder = _FakeEmbedder(
        {
            "User works on Linux.": [1.0, 0.0],
        }
    )

    matching = semantic_eval._match_snapshots(
        predicted,
        gold,
        threshold=0.80,
        embedder=embedder,
    )

    assert matching["accepted_matches"] == []
    assert matching["unmatched_predicted"] == predicted
    assert matching["unmatched_gold"] == gold


def test_match_snapshots_uses_one_to_one_matching() -> None:
    predicted = [
        _memory("projects", "active_project", "Project Alpha"),
        _memory("projects", "active_project", "Project Beta"),
    ]
    gold = [_memory("projects", "active_project", "Project Alpha canonical")]
    embedder = _FakeEmbedder(
        {
            "Project Alpha": [1.0, 0.0],
            "Project Beta": [1.0, 0.0],
            "Project Alpha canonical": [1.0, 0.0],
        }
    )

    matching = semantic_eval._match_snapshots(
        predicted,
        gold,
        threshold=0.80,
        embedder=embedder,
    )

    assert len(matching["accepted_matches"]) == 1
    assert len(matching["unmatched_predicted"]) == 1
    assert matching["unmatched_gold"] == []


def test_threshold_selection_accepts_extraction_but_rejects_overwrite() -> None:
    shared_prediction = [_memory("preferences", "reply_style", "User prefers concise bullet answers.")]
    shared_gold = [_memory("preferences", "reply_style", "User prefers concise bullet-point answers.")]
    embedder = _FakeEmbedder(
        {
            "User prefers concise bullet answers.": [1.0, 0.0],
            "User prefers concise bullet-point answers.": [0.85, 0.526782687642637],
        }
    )

    extraction_case = {
        "case_id": "extract_case",
        "level": "L0",
        "scenario": "extraction",
        "conversation": [_message("user", "remember this", "2026-04-09T13:00:00")],
        "prior_snapshot": [],
        "gold_final_snapshot": shared_gold,
        "gold_extracted_snapshot": shared_gold,
        "gold_preserved_snapshot": [],
        "notes": "",
    }
    overwrite_case = {
        **extraction_case,
        "case_id": "overwrite_case",
        "scenario": "overwrite_preservation",
        "gold_preserved_snapshot": [_memory("constraints", "tooling", "Avoid interactive git commands.")],
    }

    extraction_metrics = semantic_eval._score_case(
        case=extraction_case,
        predicted_snapshot=shared_prediction,
        threshold=semantic_eval._scenario_threshold(
            "extraction",
            extract_threshold=0.80,
            overwrite_threshold=0.88,
        ),
        embedder=embedder,
    )
    overwrite_metrics = semantic_eval._score_case(
        case=overwrite_case,
        predicted_snapshot=shared_prediction,
        threshold=semantic_eval._scenario_threshold(
            "overwrite_preservation",
            extract_threshold=0.80,
            overwrite_threshold=0.88,
        ),
        embedder=embedder,
    )

    assert extraction_metrics["true_positive_count"] == 1
    assert overwrite_metrics["true_positive_count"] == 0
    assert overwrite_metrics["false_negative_count"] == 1


def test_hybrid_scoring_uses_judge_to_rescue_paraphrase() -> None:
    case = {
        "case_id": "judge_rescue",
        "level": "L0",
        "scenario": "extraction",
        "conversation": [_message("user", "Remember this", "2026-04-09T13:00:00")],
        "prior_snapshot": [],
        "gold_final_snapshot": [_memory("personal_profile", "environment", "User works on Linux.")],
        "gold_extracted_snapshot": [_memory("personal_profile", "environment", "User works on Linux.")],
        "gold_preserved_snapshot": [],
        "notes": "",
    }
    predicted_snapshot = [
        _memory(
            "personal_profile",
            "development_environment",
            "User does most of their development work on Linux daily.",
        )
    ]
    embedder = _FakeEmbedder(
        {
            "User works on Linux.": [1.0, 0.0],
            "User does most of their development work on Linux daily.": [0.78, 0.6257795138864807],
        }
    )
    judge_provider = _ScriptedProvider(['{"verdict":"match","reason":"Same Linux environment fact."}'])

    metrics = asyncio.run(
        semantic_eval._score_case_hybrid(
            case=case,
            predicted_snapshot=predicted_snapshot,
            threshold=0.80,
            embedder=embedder,
            judge_provider=judge_provider,
            judge_model="judge-model",
        )
    )

    assert metrics["true_positive_count"] == 1
    assert metrics["false_positive_count"] == 0
    assert metrics["false_negative_count"] == 0
    assert len(metrics["semantic_matches"]) == 0
    assert len(metrics["judge_matches"]) == 1
    assert metrics["judge_matches"][0]["judge_reason_short"] == "Same Linux environment fact."
    assert metrics["semantic_unmatched_gold"] == [
        _memory("personal_profile", "environment", "User works on Linux.")
    ]
    assert metrics["semantic_unmatched_predicted"] == [
        _memory(
            "personal_profile",
            "development_environment",
            "User does most of their development work on Linux daily.",
        )
    ]


def test_hybrid_scoring_skips_judge_for_semantic_match() -> None:
    case = {
        "case_id": "semantic_match",
        "level": "L0",
        "scenario": "extraction",
        "conversation": [_message("user", "Remember Linux", "2026-04-09T13:00:00")],
        "prior_snapshot": [],
        "gold_final_snapshot": [_memory("personal_profile", "environment", "User works on Linux.")],
        "gold_extracted_snapshot": [_memory("personal_profile", "environment", "User works on Linux.")],
        "gold_preserved_snapshot": [],
        "notes": "",
    }
    predicted_snapshot = [_memory("personal_profile", "environment", "User works on Linux.")]
    embedder = _FakeEmbedder({"User works on Linux.": [1.0, 0.0]})
    judge_provider = _ScriptedProvider(['{"verdict":"match","reason":"unused"}'])

    metrics = asyncio.run(
        semantic_eval._score_case_hybrid(
            case=case,
            predicted_snapshot=predicted_snapshot,
            threshold=0.80,
            embedder=embedder,
            judge_provider=judge_provider,
            judge_model="judge-model",
        )
    )

    assert metrics["true_positive_count"] == 1
    assert len(metrics["semantic_matches"]) == 1
    assert metrics["judge_matches"] == []
    assert metrics["semantic_unmatched_gold"] == []
    assert metrics["semantic_unmatched_predicted"] == []
    assert judge_provider.call_count == 0


def test_hybrid_judge_keeps_one_to_one_matching() -> None:
    predicted_snapshot = [
        _memory("preferences", "review_style", "User wants findings first in code reviews."),
        _memory("preferences", "review_style", "User wants findings first in code review summaries."),
    ]
    gold_snapshot = [
        _memory("preferences", "review_style", "User prefers findings-first code reviews.")
    ]
    embedder = _FakeEmbedder(
        {
            "User prefers findings-first code reviews.": [1.0, 0.0],
            "User wants findings first in code reviews.": [0.75, 0.6614378277661477],
            "User wants findings first in code review summaries.": [0.7, 0.714142842854285],
        }
    )
    judge_provider = _ScriptedProvider(
        [
            '{"verdict":"match","reason":"Equivalent review preference."}',
            '{"verdict":"match","reason":"Equivalent review preference."}',
        ]
    )

    matching = asyncio.run(
        semantic_eval._hybrid_match_snapshots(
            predicted_snapshot=predicted_snapshot,
            gold_snapshot=gold_snapshot,
            scenario="extraction",
            threshold=0.95,
            embedder=embedder,
            judge_provider=judge_provider,
            judge_model="judge-model",
            judge_cache={},
        )
    )

    assert len(matching["judge_matches"]) == 1
    assert len(matching["accepted_matches"]) == 1
    assert len(matching["semantic_unmatched_predicted"]) == 2
    assert len(matching["semantic_unmatched_gold"]) == 1
    assert len(matching["unmatched_predicted"]) == 1
    assert matching["unmatched_gold"] == []


def test_hybrid_judge_respects_candidate_floor() -> None:
    predicted_snapshot = [
        _memory("preferences", "review_style", "User prefers findings-first code reviews."),
    ]
    gold_snapshot = [
        _memory("preferences", "review_style", "User prefers spoiler-free fiction recommendations."),
    ]
    embedder = _FakeEmbedder(
        {
            "User prefers findings-first code reviews.": [1.0, 0.0],
            "User prefers spoiler-free fiction recommendations.": [0.4, 0.916515138991168],
        }
    )
    judge_provider = _ScriptedProvider(['{"verdict":"match","reason":"should not happen"}'])

    matching = asyncio.run(
        semantic_eval._hybrid_match_snapshots(
            predicted_snapshot=predicted_snapshot,
            gold_snapshot=gold_snapshot,
            scenario="extraction",
            threshold=0.95,
            embedder=embedder,
            judge_provider=judge_provider,
            judge_model="judge-model",
            judge_cache={},
        )
    )

    assert matching["judge_matches"] == []
    assert judge_provider.call_count == 0


def test_run_memory_v2_semantic_eval_with_mock_provider() -> None:
    cases = [
        {
            "case_id": "extract_profile",
            "level": "L0",
            "scenario": "extraction",
            "conversation": [
                _message("user", "Remember that I work on Linux.", "2026-04-09T14:00:00"),
                _message("assistant", "Noted.", "2026-04-09T14:00:08"),
            ],
            "prior_snapshot": [],
            "gold_final_snapshot": [_memory("personal_profile", "environment", "User works on Linux.")],
            "gold_extracted_snapshot": [_memory("personal_profile", "environment", "User works on Linux.")],
            "gold_preserved_snapshot": [],
            "mock_response": {
                "history_entry": "[2026-04-09 14:00] User works on Linux.",
                "canonical_memories": [_memory("personal_profile", "environment", "User works on Linux.")],
            },
        },
        {
            "case_id": "overwrite_preserve_constraint",
            "level": "L0",
            "scenario": "overwrite_preservation",
            "conversation": [
                _message(
                    "user",
                    "Update my preference to concise bullet-point answers.",
                    "2026-04-09T14:02:00",
                ),
                _message("assistant", "Understood.", "2026-04-09T14:02:09"),
            ],
            "prior_snapshot": [
                _memory("preferences", "reply_style", "User prefers concise answers."),
                _memory("constraints", "tooling", "Avoid interactive git commands."),
            ],
            "gold_final_snapshot": [
                _memory("preferences", "reply_style", "User prefers concise bullet-point answers."),
                _memory("constraints", "tooling", "Avoid interactive git commands."),
            ],
            "gold_extracted_snapshot": [
                _memory("preferences", "reply_style", "User prefers concise bullet-point answers.")
            ],
            "gold_preserved_snapshot": [
                _memory("constraints", "tooling", "Avoid interactive git commands.")
            ],
            "mock_response": {
                "history_entry": "[2026-04-09 14:02] Reply style updated.",
                "canonical_memories": [
                    _memory("preferences", "reply_style", "User prefers concise bullet-point answers."),
                    _memory("constraints", "tooling", "Avoid interactive git commands."),
                ],
            },
        },
    ]
    embedder = _FakeEmbedder(
        {
            "User works on Linux.": [1.0, 0.0],
            "User prefers concise bullet-point answers.": [0.0, 1.0],
            "Avoid interactive git commands.": [0.7071067811865476, 0.7071067811865476],
        }
    )

    report = run_memory_v2_semantic_eval(
        cases=cases,
        provider_factory=lambda case: _ReplayProvider(case["mock_response"]),
        judge_provider=_ScriptedProvider([]),
        model="test-model",
        embedder=embedder,
    )

    assert report["evaluation"] == "memory_v2_semantic"
    assert report["total_cases"] == 2
    assert report["judge_model"] == "test-model"
    assert report["summary"]["overall"]["f1"] == pytest.approx(1.0)
    assert report["summary"]["by_scenario"]["overwrite_preservation"]["preservation_recall"] == pytest.approx(1.0)
    assert report["cases"][1]["metrics"]["dropped_preserved_items"] == []


def test_render_and_save_memory_v2_semantic_report(tmp_path) -> None:
    cases = [
        {
            "case_id": "extract_profile",
            "level": "L0",
            "scenario": "extraction",
            "conversation": [_message("user", "Remember Linux", "2026-04-09T15:00:00")],
            "prior_snapshot": [],
            "gold_final_snapshot": [_memory("personal_profile", "environment", "User works on Linux.")],
            "gold_extracted_snapshot": [_memory("personal_profile", "environment", "User works on Linux.")],
            "gold_preserved_snapshot": [],
            "mock_response": {
                "history_entry": "[2026-04-09 15:00] Linux remembered.",
                "canonical_memories": [_memory("personal_profile", "environment", "User works on Linux.")],
            },
        }
    ]
    embedder = _FakeEmbedder({"User works on Linux.": [1.0, 0.0]})

    report = run_memory_v2_semantic_eval(
        cases=cases,
        provider_factory=lambda case: _ReplayProvider(case["mock_response"]),
        judge_provider=_ScriptedProvider([]),
        model="test-model",
        embedder=embedder,
    )
    rendered = render_memory_v2_semantic_report_markdown(report)
    saved = save_memory_v2_semantic_report(report, tmp_path)

    assert "# V2 Memory Semantic Evaluation Report" in rendered
    assert "level:L0" in rendered
    assert saved["json"].exists()
    assert saved["markdown"].exists()
    assert json.loads(saved["json"].read_text(encoding="utf-8"))["total_cases"] == 1
    assert "# V2 Memory Semantic Evaluation Report" in saved["markdown"].read_text(encoding="utf-8")


def test_render_memory_v2_semantic_report_includes_missing_and_extra_items() -> None:
    report = {
        "report_version": 1,
        "evaluation": "memory_v2_semantic",
        "cases_source": "inline",
        "mode": "v2",
        "model": "test-model",
        "judge_model": "judge-model",
        "embedding_model": "fake-model",
        "total_cases": 1,
        "thresholds": {"extraction": 0.8, "overwrite_preservation": 0.88},
        "judge_candidate_floor": 0.55,
        "judge_top_k": 3,
        "summary": {
            "overall": {
                "case_count": 1,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "exact_match_rate": 0.0,
                "extraction_recall": 0.0,
                "preservation_recall": None,
                "consolidate_success_rate": 1.0,
                "raw_archive_rate": 0.0,
                "true_positive_count": 0,
                "false_positive_count": 1,
                "false_negative_count": 1,
                "extracted_gold_count": 1,
                "preserved_gold_count": 0,
            },
            "by_level": {},
            "by_scenario": {},
            "by_level_scenario": {},
        },
        "cases": [
            {
                "case_id": "mismatch_case",
                "level": "L0",
                "scenario": "extraction",
                "raw_archive_detected": False,
                "metrics": {
                    "f1": 0.0,
                    "semantic_matches": [],
                    "judge_matches": [
                        {
                            "predicted": _memory("preferences", "reply_style", "User prefers concise answers."),
                            "gold": _memory("preferences", "reply_style", "User prefers concise bullet answers."),
                            "similarity": 0.74,
                            "judge_reason_short": "Equivalent reply-style preference.",
                        }
                    ],
                    "semantic_unmatched_gold": [
                        _memory("preferences", "reply_style", "User prefers concise bullet answers.")
                    ],
                    "semantic_unmatched_predicted": [
                        _memory("preferences", "reply_style", "User prefers concise answers.")
                    ],
                    "unmatched_gold_items": [
                        _memory("preferences", "reply_style", "User prefers concise bullet-point answers.")
                    ],
                    "unmatched_predicted_items": [
                        _memory("preferences", "reply_style", "User prefers verbose answers.")
                    ],
                    "dropped_preserved_items": [],
                },
            }
        ],
    }

    rendered = render_memory_v2_semantic_report_markdown(report)

    assert "## Case Details" in rendered
    assert "### mismatch_case" in rendered
    assert "Judge Matches" in rendered
    assert "Equivalent reply-style preference." in rendered
    assert "SBERT Unmatched Gold" in rendered
    assert "[preferences/reply_style] User prefers concise bullet answers." in rendered
    assert "SBERT Unmatched Predicted" in rendered
    assert "[preferences/reply_style] User prefers concise answers." in rendered
    assert "[preferences/reply_style] User prefers concise bullet-point answers." in rendered
    assert "[preferences/reply_style] User prefers verbose answers." in rendered


def test_render_memory_v2_semantic_report_filters_case_details_to_sbert_mismatches() -> None:
    report = {
        "report_version": 1,
        "evaluation": "memory_v2_semantic",
        "cases_source": "inline",
        "mode": "v2",
        "model": "test-model",
        "judge_model": "judge-model",
        "embedding_model": "fake-model",
        "total_cases": 3,
        "thresholds": {"extraction": 0.8, "overwrite_preservation": 0.88},
        "judge_candidate_floor": 0.55,
        "judge_top_k": 3,
        "summary": {
            "overall": {
                "case_count": 3,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "exact_match_rate": 1.0,
                "extraction_recall": 1.0,
                "preservation_recall": None,
                "consolidate_success_rate": 1.0,
                "raw_archive_rate": 0.0,
                "true_positive_count": 3,
                "false_positive_count": 0,
                "false_negative_count": 0,
                "extracted_gold_count": 3,
                "preserved_gold_count": 0,
            },
            "by_level": {},
            "by_scenario": {},
            "by_level_scenario": {},
        },
        "cases": [
            {
                "case_id": "semantic_only_case",
                "level": "L0",
                "scenario": "extraction",
                "raw_archive_detected": False,
                "metrics": {
                    "f1": 1.0,
                    "semantic_matches": [
                        {
                            "main_class": "personal_profile",
                            "predicted": _memory("personal_profile", "environment", "User works on Linux."),
                            "gold": _memory("personal_profile", "environment", "User works on Linux."),
                            "similarity": 1.0,
                        }
                    ],
                    "judge_matches": [],
                    "semantic_unmatched_gold": [],
                    "semantic_unmatched_predicted": [],
                    "unmatched_gold_items": [],
                    "unmatched_predicted_items": [],
                    "dropped_preserved_items": [],
                },
            },
            {
                "case_id": "judge_rescue_case",
                "level": "L0",
                "scenario": "extraction",
                "raw_archive_detected": False,
                "metrics": {
                    "f1": 1.0,
                    "semantic_matches": [],
                    "judge_matches": [
                        {
                            "predicted": _memory("personal_profile", "development_environment", "User works on Linux daily."),
                            "gold": _memory("personal_profile", "environment", "User works on Linux."),
                            "similarity": 0.74,
                            "judge_reason_short": "Same Linux environment fact.",
                        }
                    ],
                    "semantic_unmatched_gold": [
                        _memory("personal_profile", "environment", "User works on Linux.")
                    ],
                    "semantic_unmatched_predicted": [
                        _memory("personal_profile", "development_environment", "User works on Linux daily.")
                    ],
                    "unmatched_gold_items": [],
                    "unmatched_predicted_items": [],
                    "dropped_preserved_items": [],
                },
            },
            {
                "case_id": "final_mismatch_case",
                "level": "L0",
                "scenario": "extraction",
                "raw_archive_detected": False,
                "metrics": {
                    "f1": 0.0,
                    "semantic_matches": [],
                    "judge_matches": [],
                    "semantic_unmatched_gold": [
                        _memory("preferences", "reply_style", "User prefers concise bullet answers.")
                    ],
                    "semantic_unmatched_predicted": [
                        _memory("preferences", "reply_style", "User prefers verbose answers.")
                    ],
                    "unmatched_gold_items": [
                        _memory("preferences", "reply_style", "User prefers concise bullet answers.")
                    ],
                    "unmatched_predicted_items": [
                        _memory("preferences", "reply_style", "User prefers verbose answers.")
                    ],
                    "dropped_preserved_items": [],
                },
            },
        ],
    }

    rendered = render_memory_v2_semantic_report_markdown(report)

    assert "| semantic_only_case | L0 | extraction | 1.000 |" in rendered
    assert "### semantic_only_case" not in rendered
    assert "### judge_rescue_case" in rendered
    assert "### final_mismatch_case" in rendered
