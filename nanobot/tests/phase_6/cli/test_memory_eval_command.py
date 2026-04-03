# Phase 6 test content:
# - verifies the new memory-eval CLI command emits the Phase 6 report
# - verifies report files are saved when --save-dir is provided
# - verifies invalid output modes fail fast with a clear error
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_6/cli/test_memory_eval_command.py

from __future__ import annotations

from typer.testing import CliRunner

from nanobot.cli.commands import app


runner = CliRunner()


def test_memory_eval_command_emits_json_and_saves_report(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["memory-eval", "--output", "json", "--save-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert '"report_version": 1' in result.stdout
    assert (tmp_path / "memory_phase6_ablation_report.json").exists()
    assert (tmp_path / "memory_phase6_ablation_report.md").exists()


def test_memory_eval_command_rejects_invalid_output() -> None:
    result = runner.invoke(app, ["memory-eval", "--output", "yaml"])

    assert result.exit_code == 1
    assert "--output must be 'markdown' or 'json'" in result.stdout

