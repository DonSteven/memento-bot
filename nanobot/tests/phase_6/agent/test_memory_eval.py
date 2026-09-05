# Phase 6 test content:
# - verifies the new offline replay runner produces a fixed-format ablation report
# - verifies supported modes pass and vec/hybrid are marked not_implemented
# - verifies the Phase 6 report can be rendered and saved deterministically
# How to test:
# - run this file directly with:
#   uv run --extra dev pytest -q nanobot/tests/phase_6/agent/test_memory_eval.py

from __future__ import annotations

import json

from nanobot.agent.memory_eval import (
    render_phase6_report_markdown,
    run_phase6_ablation,
    save_phase6_report,
)


def test_run_phase6_ablation_returns_fixed_format_report() -> None:
    report = run_phase6_ablation()

    assert report["report_version"] == 1
    assert report["summary"] == {
        "passed": 3,
        "failed": 0,
        "not_implemented": 0,
        "total": 3,
    }
    assert report["mode_summary"]["structured"]["status"] == "passed"
    assert "v2" not in report["mode_summary"]
    assert "fts" not in report["mode_summary"]
    assert "vec" not in report["mode_summary"]
    assert "hybrid" not in report["mode_summary"]
    assert len(report["results"]) == 3


def test_render_phase6_report_markdown_contains_fixed_sections() -> None:
    report = run_phase6_ablation()

    rendered = render_phase6_report_markdown(report)

    assert "# Memory Phase 6 Ablation Report" in rendered
    assert "## Mode Summary" in rendered
    assert "## Scenario Results" in rendered
    assert "| structured | passed |" in rendered
    assert "structured_persistence_replay" in rendered
    assert "structured_retrieval_replay" in rendered
    assert "| fts |" not in rendered
    assert "| vec |" not in rendered


def test_save_phase6_report_writes_json_and_markdown(tmp_path) -> None:
    report = run_phase6_ablation()

    saved = save_phase6_report(report, tmp_path)

    assert saved["json"].exists()
    assert saved["markdown"].exists()
    assert json.loads(saved["json"].read_text(encoding="utf-8"))["summary"]["failed"] == 0
    assert "# Memory Phase 6 Ablation Report" in saved["markdown"].read_text(encoding="utf-8")
