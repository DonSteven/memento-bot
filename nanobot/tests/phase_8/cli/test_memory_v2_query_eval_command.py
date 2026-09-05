# Phase 8 test content:
# - verifies the new memory-v2-query-eval CLI command emits the query report
# - verifies report files are saved when --save-dir is provided
# - verifies invalid argument values fail fast with a clear error
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_8/cli/test_memory_v2_query_eval_command.py

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import nanobot.agent.memory_query_eval as query_eval
from nanobot.cli import commands
from nanobot.cli.commands import app
from nanobot.config.schema import MemoryConfig

runner = CliRunner()


def _fake_runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        agents=SimpleNamespace(defaults=SimpleNamespace(model="test-model")),
        memory=MemoryConfig(vector_similarity_threshold=0.7, dynamic_top_k=8),
    )


def _fake_report() -> dict:
    return {
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
        "retrieval_config": MemoryConfig().model_dump(exclude={"embedding"}),
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
        "cases": [],
    }


def test_memory_v2_query_eval_command_emits_json_and_saves_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(commands, "_load_runtime_config", lambda: _fake_runtime_config())
    monkeypatch.setattr(commands, "_make_provider", lambda _config: object())
    monkeypatch.setattr(query_eval, "run_memory_v2_query_eval", lambda **_kwargs: _fake_report())

    result = runner.invoke(
        app,
        ["memory-v2-query-eval", "--cases-path", "synthetic-cases.json", "--output", "json", "--save-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert '"evaluation": "memory_v2_query"' in result.stdout
    assert (tmp_path / "memory_v2_query_eval_report.json").exists()
    assert (tmp_path / "memory_v2_query_eval_report.md").exists()


def test_memory_v2_query_eval_command_defaults_to_snapshot_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return _fake_report()

    monkeypatch.setattr(commands, "_load_runtime_config", lambda: _fake_runtime_config())
    monkeypatch.setattr(commands, "_make_provider", lambda _config: object())
    monkeypatch.setattr(query_eval, "run_memory_v2_query_eval", _fake_run)

    result = runner.invoke(app, ["memory-v2-query-eval", "--cases-path", "synthetic-cases.json", "--output", "json"])

    assert result.exit_code == 0
    assert captured["mode"] == "snapshot"
    assert "provider" not in captured
    assert captured["answer_provider"] is not None
    assert captured["judge_provider"] is not None
    assert captured["judge_model"] == "test-model"
    assert captured["cases_path"] == Path("synthetic-cases.json")
    assert captured["memory_config"].vector_similarity_threshold == 0.7
    assert captured["memory_config"].dynamic_top_k == 8
    assert captured["top_k"] is None


def test_memory_v2_query_eval_command_passes_provider_in_replay_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return {
            **_fake_report(),
            "mode": "replay",
        }

    monkeypatch.setattr(commands, "_load_runtime_config", lambda: _fake_runtime_config())
    monkeypatch.setattr(commands, "_make_provider", lambda _config: object())
    monkeypatch.setattr(query_eval, "run_memory_v2_query_eval", _fake_run)

    result = runner.invoke(app, ["memory-v2-query-eval", "--cases-path", "synthetic-cases.json", "--mode", "replay", "--output", "json"])

    assert result.exit_code == 0
    assert captured["mode"] == "replay"
    assert captured["provider"] is not None
    assert captured["cases_path"] == Path("synthetic-cases.json")


def test_memory_v2_query_eval_command_rejects_invalid_question_type() -> None:
    result = runner.invoke(app, ["memory-v2-query-eval", "--cases-path", "synthetic-cases.json", "--question-type", "temporal-reasoning"])

    assert result.exit_code == 1
    assert "--question-type must be" in result.stdout


def test_memory_v2_query_eval_command_rejects_invalid_mode() -> None:
    result = runner.invoke(app, ["memory-v2-query-eval", "--cases-path", "synthetic-cases.json", "--mode", "invalid"])

    assert result.exit_code == 1
    assert "--mode must be 'snapshot' or 'replay'" in result.stdout


def test_memory_v2_query_eval_command_rejects_invalid_top_k() -> None:
    result = runner.invoke(app, ["memory-v2-query-eval", "--cases-path", "synthetic-cases.json", "--top-k", "0"])

    assert result.exit_code == 1
    assert "--top-k must be greater than 0" in result.stdout


def test_query_command_requires_cases_before_loading_credentials(monkeypatch) -> None:
    def unexpected_config_load():
        raise AssertionError("configuration must not be loaded without cases")

    monkeypatch.setattr(commands, "_load_runtime_config", unexpected_config_load)
    result = runner.invoke(app, ["memory-v2-query-eval"])
    assert result.exit_code == 1
    assert "provide --cases-path" in result.stdout
