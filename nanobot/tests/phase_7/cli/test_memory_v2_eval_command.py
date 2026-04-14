# Phase 7 test content:
# - verifies the new memory-v2-eval CLI command emits the semantic report
# - verifies report files are saved when --save-dir is provided
# - verifies invalid argument values fail fast with a clear error
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_7/cli/test_memory_v2_eval_command.py

from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

import nanobot.agent.memory_semantic_eval as semantic_eval
from nanobot.cli import commands
from nanobot.cli.commands import app


runner = CliRunner()


def _fake_runtime_config() -> SimpleNamespace:
    return SimpleNamespace(agents=SimpleNamespace(defaults=SimpleNamespace(model="test-model")))


def _fake_report() -> dict:
    return {
        "report_version": 1,
        "evaluation": "memory_v2_semantic",
        "cases_source": "inline",
        "mode": "v2",
        "model": "test-model",
        "judge_model": "test-model",
        "embedding_model": "fake-model",
        "total_cases": 1,
        "thresholds": {"extraction": 0.8, "overwrite_preservation": 0.88},
        "judge_candidate_floor": 0.55,
        "judge_top_k": 3,
        "summary": {
            "overall": {
                "case_count": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "exact_match_rate": 1.0,
                "extraction_recall": 1.0,
                "preservation_recall": None,
                "consolidate_success_rate": 1.0,
                "raw_archive_rate": 0.0,
                "true_positive_count": 1,
                "false_positive_count": 0,
                "false_negative_count": 0,
                "extracted_gold_count": 1,
                "preserved_gold_count": 0,
            },
            "by_level": {},
            "by_scenario": {},
            "by_level_scenario": {},
        },
        "cases": [
            {
                "case_id": "case_1",
                "level": "L0",
                "scenario": "extraction",
                "raw_archive_detected": False,
                "metrics": {
                    "f1": 1.0,
                    "semantic_matches": [],
                    "judge_matches": [],
                    "unmatched_gold_items": [],
                    "unmatched_predicted_items": [],
                    "dropped_preserved_items": [],
                },
            }
        ],
    }


def test_memory_v2_eval_command_emits_json_and_saves_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(commands, "_load_runtime_config", lambda: _fake_runtime_config())
    monkeypatch.setattr(commands, "_make_provider", lambda _config: object())
    monkeypatch.setattr(semantic_eval, "run_memory_v2_semantic_eval", lambda **_kwargs: _fake_report())

    result = runner.invoke(
        app,
        ["memory-v2-eval", "--output", "json", "--save-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert '"evaluation": "memory_v2_semantic"' in result.stdout
    assert (tmp_path / "memory_v2_semantic_eval_report.json").exists()
    assert (tmp_path / "memory_v2_semantic_eval_report.md").exists()


def test_memory_v2_eval_command_defaults_judge_model_to_main_model(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return _fake_report()

    monkeypatch.setattr(commands, "_load_runtime_config", lambda: _fake_runtime_config())
    monkeypatch.setattr(commands, "_make_provider", lambda _config: object())
    monkeypatch.setattr(semantic_eval, "run_memory_v2_semantic_eval", _fake_run)

    result = runner.invoke(app, ["memory-v2-eval", "--output", "json"])

    assert result.exit_code == 0
    assert captured["judge_model"] == "test-model"
    assert captured["judge_provider"] is not None


def test_memory_v2_eval_command_passes_explicit_judge_model(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return {
            **_fake_report(),
            "judge_model": "judge-model",
        }

    monkeypatch.setattr(commands, "_load_runtime_config", lambda: _fake_runtime_config())
    monkeypatch.setattr(commands, "_make_provider", lambda _config: object())
    monkeypatch.setattr(semantic_eval, "run_memory_v2_semantic_eval", _fake_run)

    result = runner.invoke(app, ["memory-v2-eval", "--judge-model", "judge-model", "--output", "json"])

    assert result.exit_code == 0
    assert captured["judge_model"] == "judge-model"


def test_memory_v2_eval_command_rejects_invalid_level() -> None:
    result = runner.invoke(app, ["memory-v2-eval", "--level", "L9"])

    assert result.exit_code == 1
    assert "--level must be 'L0', 'L1', or 'all'" in result.stdout


def test_memory_v2_eval_command_rejects_invalid_scenario() -> None:
    result = runner.invoke(app, ["memory-v2-eval", "--scenario", "overwrite"])

    assert result.exit_code == 1
    assert "--scenario must be 'extraction', 'overwrite_preservation', or 'all'" in result.stdout
