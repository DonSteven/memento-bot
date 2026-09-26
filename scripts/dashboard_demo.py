"""Serve the real Dashboard with isolated, synthetic observations.

Standalone: .venv/bin/python scripts/dashboard_demo.py --data-dir /tmp/new-demo-dir
Capture mode is driven over stdin by capture_dashboard_demo.mjs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web

from nanobot.agent.knowledge_retrieval import KnowledgeResult
from nanobot.api.dashboard import create_dashboard_app
from nanobot.observability.events import DashboardEvents
from nanobot.observability.hook import preview
from nanobot.observability.store import ObservabilityStore


FIXTURE_IDS = ("demo-complete-03", "demo-error-02", "demo-complete-01")
LIVE_ID = "demo-live-04"


class FakeMemory:
    database = SimpleNamespace(read_snapshot=lambda: SimpleNamespace(revision=0, memories=[]))
    config = SimpleNamespace(dynamic_top_k=5)
    embedder = object()

    async def sync_markdown(self):
        return None

    async def search_dynamic(self, query, limit):
        return []


class FakeKnowledge:
    async def retrieve(self, query, **_kwargs):
        return KnowledgeResult(query=query, status="insufficient", sufficient=False,
                               reason="Synthetic demo has no external evidence.",
                               missing_points=("No demo source pages",), parents=(), children=())


class FakeCron:
    def list_jobs(self, include_disabled=False):
        return []


def prepare_store(destination: Path) -> ObservabilityStore:
    destination = destination.resolve()
    if not destination.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError("--data-dir must be under the system temporary directory")
    if destination.exists():
        raise ValueError(f"refusing to reuse existing directory: {destination}")
    destination.mkdir(parents=True)
    store = ObservabilityStore(destination / "dashboard.db")
    store.initialize()
    return store


def seed(store: ObservabilityStore, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    rows = (
        (FIXTURE_IDS[0], 8, "completed", "completed", 1360),
        (FIXTURE_IDS[1], 190, "error", "error", 1480),
        (FIXTURE_IDS[2], 480, "completed", "completed", 1850),
    )
    for run_id, minutes_ago, status, reason, duration in rows:
        store.start_run(run_id, "cli:synthetic-demo", "cli", "demo-model",
                        (now - timedelta(minutes=minutes_ago)).isoformat())
        store.add_event(run_id, "memory", {"core_count": 2, "retrieved_count": 1},
                        duration_ms=18)
        store.add_event(run_id, "model", {
            "usage": {"prompt_tokens": 120, "completion_tokens": 32},
            "usage_reported": True, "stop_reason": "tool_calls", "tool_call_count": 1,
            "content_preview": preview("Synthetic project: checking the release checklist."),
        }, iteration=0, duration_ms=520)
        store.add_event(run_id, "tools", {"calls": [{
            "call_id": "demo-call-1", "name": "kb_search",
            "arguments": preview({"query": "synthetic release checklist"}),
            "status": "ok" if status == "completed" else "error",
            "result_preview": preview("Synthetic note: tests, docs, and dashboard assets are ready."),
        }]}, iteration=0, duration_ms=260)
        store.add_event(run_id, "model", {
            "usage": {"prompt_tokens": 170, "completion_tokens": 58},
            "usage_reported": True, "stop_reason": reason, "tool_call_count": 0,
            "content_preview": preview("Synthetic answer: the checklist was reviewed."),
        }, iteration=1, duration_ms=430)
        store.add_usage(run_id, 290, 90, True)
        store.finish_run(run_id, status, reason, duration,
                         "Synthetic provider error" if status == "error" else None)


class DemoReplay:
    STEPS = ("start", "memory", "model", "tools", "final_model", "finish")

    def __init__(self, store: ObservabilityStore, events: DashboardEvents):
        self.store = store
        self.events = events
        self.position = 0

    def step(self, command: str) -> dict:
        expected = self.STEPS[self.position] if self.position < len(self.STEPS) else "stop"
        if command != expected:
            raise ValueError(f"expected {expected}, got {command}")
        if command == "start":
            self.store.start_run(LIVE_ID, "cli:synthetic-demo", "cli", "demo-model")
            event_type = "run.started"
        elif command == "memory":
            self.store.add_event(LIVE_ID, "memory", {"core_count": 2, "retrieved_count": 1},
                                 duration_ms=18)
            event_type = "run.updated"
        elif command == "model":
            self.store.add_event(LIVE_ID, "model", {
                "usage": {"prompt_tokens": 120, "completion_tokens": 32},
                "usage_reported": True, "stop_reason": "tool_calls", "tool_call_count": 1,
                "content_preview": preview("Synthetic project: checking the release checklist."),
            }, iteration=0, duration_ms=520)
            self.store.add_usage(LIVE_ID, 120, 32, True)
            event_type = "run.updated"
        elif command == "tools":
            self.store.add_event(LIVE_ID, "tools", {"calls": [{
                "call_id": "demo-live-call", "name": "kb_search",
                "arguments": preview({"query": "synthetic release checklist"}),
                "status": "ok",
                "result_preview": preview("Synthetic note: tests, docs, and dashboard assets are ready."),
            }]}, iteration=0, duration_ms=260)
            event_type = "run.updated"
        elif command == "final_model":
            self.store.add_event(LIVE_ID, "model", {
                "usage": {"prompt_tokens": 170, "completion_tokens": 58},
                "usage_reported": True, "stop_reason": "completed", "tool_call_count": 0,
                "content_preview": preview("Synthetic answer: the checklist was reviewed."),
            }, iteration=1, duration_ms=430)
            self.store.add_usage(LIVE_ID, 170, 58, True)
            event_type = "run.updated"
        else:
            self.store.finish_run(LIVE_ID, "completed", "completed", 1360)
            event_type = "run.finished"
        self.position += 1
        self.events.publish(event_type, run_id=LIVE_ID)
        detail = self.store.get_run(LIVE_ID)
        return {"type": "step", "step": command, "run_id": LIVE_ID,
                "status": detail["run"]["status"], "events": len(detail["events"]),
                "prompt_tokens": detail["run"]["prompt_tokens"],
                "completion_tokens": detail["run"]["completion_tokens"]}


async def capture_server(app: web.Application, replay: DemoReplay) -> None:
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        print(json.dumps({"type": "ready", "url": f"http://127.0.0.1:{port}/dashboard/",
                          "fixture_ids": FIXTURE_IDS, "live_id": LIVE_ID}), flush=True)
        while line := await asyncio.to_thread(sys.stdin.readline):
            command = line.strip()
            if command == "stop":
                print(json.dumps({"type": "stopped"}), flush=True)
                break
            try:
                result = replay.step(command)
            except ValueError as exc:
                print(json.dumps({"type": "error", "message": str(exc)}), flush=True)
                raise
            print(json.dumps(result), flush=True)
    finally:
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="new directory under the system temporary directory")
    parser.add_argument("--port", type=int, default=18791,
                        help="standalone listener port (capture mode chooses an available port)")
    parser.add_argument("--capture", action="store_true", help="stdin-controlled capture mode")
    args = parser.parse_args()
    try:
        store = prepare_store(args.data_dir)
    except ValueError as exc:
        parser.error(str(exc))
    seed(store)
    events = DashboardEvents()
    agent = SimpleNamespace(memory_service=FakeMemory(), knowledge_retriever=FakeKnowledge())
    app = create_dashboard_app(agent, FakeCron(), store, events=events)
    if args.capture:
        asyncio.run(capture_server(app, DemoReplay(store, events)))
    else:
        print(f"Synthetic demo data at {args.data_dir}; open http://127.0.0.1:{args.port}/dashboard/")
        web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
