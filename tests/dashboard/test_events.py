from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.observability.events import DashboardEvents


def test_two_subscribers_and_slow_client():
    events = DashboardEvents(queue_size=2)
    fast = events.subscribe()
    slow = events.subscribe()
    events.publish("run.started", run_id="r1")
    assert fast.get_nowait() == {"type": "run.started", "run_id": "r1"}
    events.publish("run.updated", run_id="r1")
    assert fast.get_nowait() == {"type": "run.updated", "run_id": "r1"}
    events.publish("run.finished", run_id="r1")
    assert fast.get_nowait() == {"type": "run.finished", "run_id": "r1"}
    assert slow.get_nowait() is None
    assert slow not in events._subscribers
    events.close()
    assert fast.get_nowait() is None


def test_unsubscribe_and_content_free_events():
    events = DashboardEvents()
    first = events.subscribe()
    second = events.subscribe()
    events.unsubscribe(first)
    events.publish("task.changed", job_id="j1")
    assert first.empty()
    assert second.get_nowait() == {"type": "task.changed", "job_id": "j1"}


def test_cron_notification_failure_does_not_undo_saved_job(tmp_path):
    def fail(_job_id):
        raise RuntimeError("subscriber failed")

    cron = CronService(tmp_path / "jobs.json", on_change=fail)
    job = cron.add_job("Safe", CronSchedule(kind="every", every_ms=60000), "work")
    assert CronService(tmp_path / "jobs.json").get_job(job.id) is not None
