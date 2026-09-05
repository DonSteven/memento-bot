# Phase 8 test content:
# - verifies snapshot-first query fixtures load without requiring the source dataset
# - verifies support-memory matching respects main_class buckets and semantic thresholds
# - verifies snapshot-seeded retrieval scoring, stale detection, and answer correctness
# - verifies replay mode remains available as a diagnostic path
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_8/agent/test_memory_query_eval.py

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_query_eval as query_eval
from nanobot.agent.memory_query_eval import (
    load_memory_query_cases,
    render_memory_v2_query_report_markdown,
    run_memory_v2_query_eval,
    save_memory_v2_query_report,
)
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]


class _ScriptedProvider(LLMProvider):
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
        return LLMResponse(content="")

    def get_default_model(self) -> str:
        return "test-model"


def _tool_response(arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="save_memory_structured_call",
                name="save_memory_structured",
                arguments=arguments,
            )
        ],
    )


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

    monkeypatch.setattr(query_eval, "_load_hungarian", lambda: _fake_linear_sum_assignment)


def _memory(main_class: str, sub_class: str, text: str) -> dict[str, str]:
    return {"main_class": main_class, "sub_class": sub_class, "text": text}


def _write_mode_fixture(tmp_path: Path, mode: str) -> Path:
    """Create fictional unit-test cases without external benchmark records."""
    mode_root = tmp_path / mode / "l0"
    mode_root.mkdir(parents=True, exist_ok=True)
    cases_path = mode_root / "cases.json"
    if mode == "snapshot":
        payload = {
            "cases": [
                {
                    "case_id": "user_degree",
                    "level": "L0",
                    "source_question_id": "qid_user",
                    "source_question_type": "single-session-user",
                    "question": "What degree did I graduate with?",
                    "question_date": "2023/05/30 (Tue) 23:40",
                    "gold_answer": "Applied Cartography",
                    "gold_answer_aliases": [],
                    "canonical_snapshot": [
                        _memory(
                            "personal_profile",
                            "education",
                            "User graduated with a degree in Applied Cartography.",
                        ),
                    ],
                    "gold_support_memories": [
                        _memory(
                            "personal_profile",
                            "education",
                            "User graduated with a degree in Applied Cartography.",
                        ),
                    ],
                    "gold_forbidden_memories": [],
                },
                {
                    "case_id": "update_rowing",
                    "level": "L0",
                    "source_question_id": "qid_update",
                    "source_question_type": "knowledge-update",
                    "question": "What is my personal best time in the indoor rowing sprint?",
                    "question_date": "2023/05/30 (Tue) 23:40",
                    "gold_answer": "18 minutes and 40 seconds",
                    "gold_answer_aliases": ["18:40"],
                    "canonical_snapshot": [
                        _memory(
                            "daily_life",
                            "fitness_record",
                            "User's personal best time in the indoor rowing sprint is 18 minutes and 40 seconds.",
                        ),
                    ],
                    "gold_support_memories": [
                        _memory(
                            "daily_life",
                            "fitness_record",
                            "User's personal best time in the indoor rowing sprint is 18 minutes and 40 seconds.",
                        )
                    ],
                    "gold_forbidden_memories": [],
                },
            ]
        }
    else:
        payload = {
            "cases": [
                {
                    "case_id": "update_rowing",
                    "level": "L0",
                    "source_question_id": "qid_update",
                    "source_question_type": "knowledge-update",
                    "question": "What is my personal best time in the indoor rowing sprint?",
                    "question_date": "2023/05/30 (Tue) 23:40",
                    "gold_answer": "18 minutes and 40 seconds",
                    "gold_answer_aliases": ["18:40"],
                    "canonical_snapshot": [
                        _memory(
                            "daily_life",
                            "fitness_record",
                            "User's personal best time in the indoor rowing sprint is 18 minutes and 40 seconds.",
                        ),
                    ],
                    "gold_support_memories": [
                        _memory(
                            "daily_life",
                            "fitness_record",
                            "User's personal best time in the indoor rowing sprint is 18 minutes and 40 seconds.",
                        )
                    ],
                    "gold_forbidden_memories": [
                        _memory(
                            "daily_life",
                            "fitness_record",
                            "User's personal best time in the indoor rowing sprint is 21 minutes and 15 seconds.",
                        )
                    ],
                    "replay_sessions": [
                        {
                            "session_id": "sess_old",
                            "session_date": "2023/05/20 (Sat) 02:21",
                            "conversation": [
                                {
                                    "role": "user",
                                    "content": "I recently set a personal best time in a indoor rowing sprint with a time of 21:15.",
                                    "timestamp": "2023-05-20T02:21:00",
                                },
                                {
                                    "role": "assistant",
                                    "content": "21:15 is a strong baseline to build on.",
                                    "timestamp": "2023-05-20T02:21:15",
                                },
                            ],
                        },
                        {
                            "session_id": "sess_new",
                            "session_date": "2023/05/30 (Tue) 21:40",
                            "conversation": [
                                {
                                    "role": "user",
                                    "content": "I'm hoping to beat my personal best time of 18:40 this time around.",
                                    "timestamp": "2023-05-30T21:40:00",
                                },
                                {
                                    "role": "assistant",
                                    "content": "18:40 is a great target and current best.",
                                    "timestamp": "2023-05-30T21:40:15",
                                },
                            ],
                        },
                    ],
                }
            ]
        }
    cases_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return tmp_path / mode


def test_load_memory_query_cases_accepts_snapshot_fixtures_without_source_dataset(tmp_path: Path) -> None:
    cases_root = _write_mode_fixture(tmp_path, "snapshot")

    all_cases = load_memory_query_cases(cases_root, mode="snapshot")
    update_cases = load_memory_query_cases(cases_root, mode="snapshot", question_type="knowledge-update")
    user_cases = load_memory_query_cases(cases_root, mode="snapshot", question_type="single-session-user")

    assert len(all_cases) == 2
    assert len(update_cases) == 1
    assert update_cases[0]["source_question_type"] == "knowledge-update"
    assert update_cases[0]["gold_forbidden_memories"] == []
    assert len(update_cases[0]["canonical_snapshot"]) == 1
    assert len(user_cases) == 1
    assert all_cases[0]["canonical_snapshot"][0]["main_class"] == "personal_profile"


def test_snapshot_fixtures_keep_knowledge_updates_latest_only(tmp_path: Path) -> None:
    cases = load_memory_query_cases(_write_mode_fixture(tmp_path, "snapshot"), mode="snapshot", question_type="knowledge-update")

    assert cases
    for case in cases:
        assert case["gold_forbidden_memories"] == []
        assert len(case["canonical_snapshot"]) == 1


def test_seed_snapshot_writes_db_and_views(tmp_path: Path) -> None:
    snapshot = [
        _memory("personal_profile", "education", "User graduated with a degree in Applied Cartography."),
        _memory("preferences", "reply_style", "User prefers concise answers."),
    ]

    seeded = query_eval._seed_snapshot(tmp_path, snapshot)
    db = query_eval.MemoryDatabase(tmp_path)

    assert seeded == snapshot
    assert db.read_core_memories()[0].text == snapshot[0]["text"]
    assert "User prefers concise answers." in db.memory_file.read_text(encoding="utf-8")


def test_match_memory_snapshots_supports_semantic_matching() -> None:
    predicted = [_memory("preferences", "gear", "User prefers Sony-compatible accessories for their camera setup.")]
    gold = [_memory("preferences", "gear", "User prefers Sony-compatible accessories or high-quality photography gear for their photography setup.")]
    embedder = _FakeEmbedder(
        {
            predicted[0]["text"]: [1.0, 0.0],
            gold[0]["text"]: [1.0, 0.0],
        }
    )

    matching = query_eval._match_memory_snapshots(
        predicted,
        gold,
        threshold=0.82,
        embedder=embedder,
    )

    assert len(matching["accepted_matches"]) == 1
    assert matching["accepted_matches"][0]["match_stage"] == "semantic"
    assert matching["unmatched_gold"] == []
    assert matching["unmatched_predicted"] == []


def test_deterministic_answer_match_accepts_aliases() -> None:
    assert query_eval._deterministic_answer_match(
        "18:40",
        "18 minutes and 40 seconds",
        ["18:40"],
    )


def test_run_memory_v2_query_eval_snapshot_mode_scores_retrieval_answer_without_stale(tmp_path: Path) -> None:
    cases_root = _write_mode_fixture(tmp_path, "snapshot")
    cases = load_memory_query_cases(cases_root, mode="snapshot")

    embedder = _FakeEmbedder(
        {
            "User graduated with a degree in Applied Cartography.": [1.0, 0.0],
            "User's personal best time in the indoor rowing sprint is 18 minutes and 40 seconds.": [0.0, 1.0],
            "User's personal best time in the indoor rowing sprint is 21 minutes and 15 seconds.": [0.0, 0.9],
            "User prefers concise answers.": [0.5, 0.5],
        }
    )

    def _answer_provider(case: dict[str, Any]) -> _ScriptedProvider:
        if case["case_id"] == "user_degree":
            return _ScriptedProvider([LLMResponse(content="He graduated in applied cartography.")])
        return _ScriptedProvider([LLMResponse(content="18:40")])

    def _judge_provider(case: dict[str, Any]) -> _ScriptedProvider:
        return _ScriptedProvider([LLMResponse(content='{"verdict":"correct"}')])

    report = run_memory_v2_query_eval(
        mode="snapshot",
        cases=cases,
        answer_provider_factory=_answer_provider,
        judge_provider_factory=_judge_provider,
        model="test-model",
        judge_model="judge-model",
        embedder=embedder,
    )

    assert report["evaluation"] == "memory_v2_query"
    assert report["mode"] == "snapshot"
    assert report["total_cases"] == 2
    assert report["summary"]["overall"]["answer_accuracy"] == pytest.approx(1.0)
    assert report["summary"]["overall"]["answer_judge_pass_rate"] == pytest.approx(0.5)
    assert report["summary"]["overall"]["support_hit_at_k"] == pytest.approx(0.5)
    assert report["summary"]["overall"]["core_only_rescue_rate"] == pytest.approx(0.5)
    assert report["summary"]["overall"]["stale_exposed_rate"] is None
    assert report["cases"][0]["metrics"]["answer_score_method"] == "judge"
    assert report["cases"][0]["retrieval_source_mode"] == "snapshot"
    assert report["cases"][1]["metrics"]["answer_score_method"] == "exact"
    assert report["cases"][1]["stale_retrieved_memories"] == []


def test_run_memory_v2_query_eval_replay_mode_smoke(tmp_path: Path) -> None:
    cases_root = _write_mode_fixture(tmp_path, "replay")
    replay_case = load_memory_query_cases(cases_root, mode="replay")

    embedder = _FakeEmbedder(
        {
            "User's personal best time in the indoor rowing sprint is 18 minutes and 40 seconds.": [1.0, 0.0],
            "User's personal best time in the indoor rowing sprint is 21 minutes and 15 seconds.": [0.0, 1.0],
        }
    )

    def _consolidation_provider(_case: dict[str, Any]) -> _ScriptedProvider:
        return _ScriptedProvider(
            [
                _tool_response(
                    {
                        "history_entry": "[2023-05-20 02:21] User's personal best time in the indoor rowing sprint is 21 minutes and 15 seconds.",
                        "canonical_memories": [
                            _memory(
                                "daily_life",
                                "fitness_record",
                                "User's personal best time in the indoor rowing sprint is 21 minutes and 15 seconds.",
                            )
                        ],
                    }
                ),
                _tool_response(
                    {
                        "history_entry": "[2023-05-30 21:40] User's personal best time in the indoor rowing sprint is 18 minutes and 40 seconds.",
                        "canonical_memories": [
                            _memory(
                                "daily_life",
                                "fitness_record",
                                "User's personal best time in the indoor rowing sprint is 21 minutes and 15 seconds.",
                            ),
                            _memory(
                                "daily_life",
                                "fitness_record",
                                "User's personal best time in the indoor rowing sprint is 18 minutes and 40 seconds.",
                            ),
                        ],
                    }
                ),
            ]
        )

    report = run_memory_v2_query_eval(
        mode="replay",
        cases=replay_case,
        provider_factory=_consolidation_provider,
        answer_provider=_ScriptedProvider([LLMResponse(content="18:40")]),
        judge_provider=_ScriptedProvider([LLMResponse(content='{"verdict":"correct"}')]),
        model="test-model",
        judge_model="judge-model",
        embedder=embedder,
    )

    assert report["mode"] == "replay"
    assert report["cases"][0]["retrieval_source_mode"] == "replay"
    assert report["cases"][0]["replay_diagnostics"]["consolidate_success_rate"] == pytest.approx(1.0)
    assert report["summary"]["overall"]["stale_exposed_rate"] == pytest.approx(1.0)


def test_render_and_save_query_report(tmp_path: Path) -> None:
    report = {
        "report_version": 1,
        "evaluation": "memory_v2_query",
        "cases_source": "inline",
        "source_dataset": None,
        "mode": "snapshot",
        "memory_mode": "v2",
        "model": "test-model",
        "judge_model": "judge-model",
        "embedding_model": "fake-model",
        "top_k": 5,
        "threshold": 0.82,
        "total_cases": 1,
        "summary": {
            "overall": {
                "case_count": 1,
                "support_hit_at_1": 1.0,
                "support_hit_at_k": 1.0,
                "support_recall_at_k": 1.0,
                "support_mrr": 1.0,
                "stale_exposed_rate": None,
                "retrieval_exact_match_rate": 1.0,
                "answer_accuracy": 1.0,
                "answer_exact_rate": 1.0,
                "answer_judge_pass_rate": 0.0,
                "unsupported_but_correct_rate": 0.0,
                "core_only_rescue_rate": 0.0,
                "retrieval_answer_consistency": {
                    "hit_correct": 1,
                    "hit_wrong": 0,
                    "miss_correct": 0,
                    "miss_wrong": 0,
                },
            },
            "by_level": {},
            "by_question_type": {},
        },
        "cases": [
            {
                "case_id": "case_1",
                "level": "L0",
                "source_question_type": "single-session-user",
                "question": "Q?",
                "gold_answer": "A",
                "generated_answer": "A",
                "canonical_snapshot": [_memory("personal_profile", "education", "A")],
                "support_available_in_retrieved": True,
                "support_available_in_core": True,
                "missing_support_memories": [],
                "stale_retrieved_memories": [],
                "retrieval_source_mode": "snapshot",
                "metrics": {
                    "stale_exposed": False,
                    "answer_accuracy": True,
                    "answer_score_method": "exact",
                },
            }
        ],
    }

    rendered = render_memory_v2_query_report_markdown(report)
    saved = save_memory_v2_query_report(report, tmp_path)

    assert "# V2 Memory Query Evaluation Report" in rendered
    assert "Retrieval vs Answer" in rendered
    assert "Retrieval Source Mode" in rendered
    assert saved["json"].exists()
    assert saved["markdown"].exists()
    assert json.loads(saved["json"].read_text(encoding="utf-8"))["evaluation"] == "memory_v2_query"


def test_query_cases_require_an_explicit_dataset_path() -> None:
    with pytest.raises(ValueError, match="datasets are not bundled"):
        load_memory_query_cases()
