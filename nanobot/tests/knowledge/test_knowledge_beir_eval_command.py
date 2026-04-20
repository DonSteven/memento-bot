# Knowledge BEIR CLI test content:
# - verifies the CLI emits JSON and saves report files
# - verifies the command forwards the expected benchmark arguments
# - verifies invalid dataset and output values fail fast

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import nanobot.agent.knowledge_beir_eval as knowledge_beir_eval
from nanobot.cli import commands
from nanobot.cli.commands import app

runner = CliRunner()


def _fake_runtime_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        agents=SimpleNamespace(defaults=SimpleNamespace(model="test-model")),
        workspace_path=tmp_path,
    )


def _fake_report() -> dict:
    return {
        "report_version": 2,
        "evaluation": "knowledge_beir",
        "dataset": "scifact",
        "bench_dir": "/tmp/bench",
        "dataset_path": "/tmp/bench/datasets/scifact",
        "model": "test-model",
        "embedding_model": "fake-embedding-model",
        "doc_limit": 10,
        "evidence_limit": 5,
        "corpus_size": 22,
        "query_count": 2,
        "ingestion_summary": {
            "inserted": 22,
            "updated": 0,
            "unchanged": 0,
            "failed": 0,
        },
        "summary": {
            "ndcg": {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0, 20: 0.95, 50: 0.92, 100: 0.9},
            "map": {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0, 20: 0.95, 50: 0.92, 100: 0.9},
            "recall": {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0, 20: 0.95, 50: 0.92, 100: 0.9},
            "precision": {1: 1.0, 3: 0.5, 5: 0.4, 10: 0.2, 20: 0.15, 50: 0.12, 100: 0.1},
            "mrr": {10: 1.0, 20: 1.0, 50: 1.0, 100: 1.0},
        },
        "diagnostics": {
            "average_candidate_parents": 10.5,
            "average_evidence_children": 2.5,
            "relevant_doc_in_candidate_parents_rate": 1.0,
            "relevant_doc_in_evidence_rate": 1.0,
        },
        "cases": [],
    }


def test_knowledge_beir_eval_command_emits_json_and_saves_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(commands, "_load_runtime_config", lambda: _fake_runtime_config(tmp_path))
    monkeypatch.setattr(commands, "_make_provider", lambda _config: object())
    monkeypatch.setattr(knowledge_beir_eval, "run_knowledge_beir_eval", lambda **_kwargs: _fake_report())

    result = runner.invoke(
        app,
        ["knowledge-beir-eval", "--output", "json", "--save-dir", str(tmp_path / "reports")],
    )

    assert result.exit_code == 0
    assert '"evaluation": "knowledge_beir"' in result.stdout
    assert (tmp_path / "reports" / "knowledge_beir_eval_report.json").exists()
    assert (tmp_path / "reports" / "knowledge_beir_eval_report.md").exists()


def test_knowledge_beir_eval_command_passes_expected_arguments(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return _fake_report()

    monkeypatch.setattr(commands, "_load_runtime_config", lambda: _fake_runtime_config(tmp_path))
    monkeypatch.setattr(commands, "_make_provider", lambda _config: object())
    monkeypatch.setattr(knowledge_beir_eval, "run_knowledge_beir_eval", _fake_run)

    result = runner.invoke(app, ["knowledge-beir-eval", "--output", "json"])

    assert result.exit_code == 0
    assert captured["dataset"] == "scifact"
    assert captured["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["bench_dir"] == tmp_path / "benchmarks" / "beir" / "scifact"
    assert "summary_model" not in captured


def test_knowledge_beir_eval_command_rejects_invalid_dataset() -> None:
    result = runner.invoke(app, ["knowledge-beir-eval", "--dataset", "nfcorpus"])

    assert result.exit_code == 1
    assert "--dataset must be 'scifact'" in result.stdout


def test_knowledge_beir_eval_command_rejects_invalid_output() -> None:
    result = runner.invoke(app, ["knowledge-beir-eval", "--output", "yaml"])

    assert result.exit_code == 1
    assert "--output must be 'markdown' or 'json'" in result.stdout
