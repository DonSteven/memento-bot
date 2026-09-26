"""The synthetic replay persists each state before broadcasting it."""

import pytest

from nanobot.observability.events import DashboardEvents
from scripts.dashboard_demo import DemoReplay, FIXTURE_IDS, LIVE_ID, prepare_store, seed


def test_replay_persists_before_notification_and_finishes_once(tmp_path):
    store = prepare_store(tmp_path / "new-demo")
    events = DashboardEvents()
    subscriber = events.subscribe()
    replay = DemoReplay(store, events)
    assert store.get_run(LIVE_ID) is None

    for command, event_type, event_count, usage in (
        ("start", "run.started", 0, (0, 0)),
        ("memory", "run.updated", 1, (0, 0)),
        ("model", "run.updated", 2, (120, 32)),
        ("tools", "run.updated", 3, (120, 32)),
        ("final_model", "run.updated", 4, (290, 90)),
        ("finish", "run.finished", 4, (290, 90)),
    ):
        result = replay.step(command)
        detail = store.get_run(LIVE_ID)
        assert subscriber.get_nowait() == {"type": event_type, "run_id": LIVE_ID}
        assert result["run_id"] == detail["run"]["run_id"] == LIVE_ID
        assert result["events"] == len(detail["events"]) == event_count
        assert (detail["run"]["prompt_tokens"], detail["run"]["completion_tokens"]) == usage
        assert detail["run"]["status"] == ("completed" if command == "finish" else "running")

    model_usage = [event["data"]["usage"] for event in detail["events"] if event["kind"] == "model"]
    assert sum(item["prompt_tokens"] for item in model_usage) == 290
    assert sum(item["completion_tokens"] for item in model_usage) == 90
    assert detail["run"]["duration_ms"] >= sum(
        event["duration_ms"] for event in detail["events"]
    )
    with pytest.raises(ValueError, match="expected stop, got finish"):
        replay.step("finish")
    assert subscriber.empty()


def test_replay_rejects_out_of_order_and_existing_directory(tmp_path):
    store = prepare_store(tmp_path / "new-demo")
    events = DashboardEvents()
    subscriber = events.subscribe()
    replay = DemoReplay(store, events)
    with pytest.raises(ValueError, match="expected start, got tools"):
        replay.step("tools")
    assert store.get_run(LIVE_ID) is None
    assert subscriber.empty()
    with pytest.raises(ValueError, match="refusing to reuse existing directory"):
        prepare_store(tmp_path / "new-demo")


def test_seeded_runs_cover_their_sequential_event_durations(tmp_path):
    store = prepare_store(tmp_path / "new-demo")
    seed(store)
    for run_id in FIXTURE_IDS:
        detail = store.get_run(run_id)
        assert detail["run"]["duration_ms"] >= sum(
            event["duration_ms"] for event in detail["events"]
        )
