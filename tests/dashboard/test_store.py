from datetime import datetime, timedelta, timezone

from nanobot.observability.store import ObservabilityStore


def test_store_pagination_overview_and_events(tmp_path):
    store = ObservabilityStore(tmp_path / "observability" / "dashboard.db")
    store.initialize()
    now = datetime.now(timezone.utc)
    for index in range(3):
        run_id = f"run-{index}"
        store.start_run(run_id, "cli:test", "cli", "test-model", now.isoformat())
        store.add_event(run_id, "tools", {"calls": [{"status": "error"}, {"status": "ok"}]})
        store.add_usage(run_id, 10, 2, True)
        store.finish_run(run_id, "completed" if index else "stopped",
                         "completed" if index else "max_iterations", 10 + index)
    first = store.list_runs(limit=2)
    second = store.list_runs(limit=2, cursor=first["next_cursor"])
    assert len(first["items"]) == 2
    assert len(second["items"]) == 1
    assert len({item["run_id"] for item in first["items"] + second["items"]}) == 3
    assert store.get_run("run-0")["events"][0]["data"]["calls"][0]["status"] == "error"
    assert store.get_run("missing") is None
    other = ObservabilityStore(store.path)
    assert other.get_run("run-0")["run"]["prompt_tokens"] == 10
    overview = store.get_overview(now + timedelta(microseconds=1))
    assert overview["runs"] == 3
    assert overview["success_rate"] == 2 / 3
    assert overview["p95_latency_ms"] == 12
    assert overview["tool_errors"] == 3
    assert len(overview["hours"]) == 24


def test_empty_overview_is_not_false_zero(tmp_path):
    store = ObservabilityStore(tmp_path / "dashboard.db")
    store.initialize()
    overview = store.get_overview()
    assert overview["success_rate"] is None
    assert overview["avg_latency_ms"] is None
    assert overview["p95_latency_ms"] is None
