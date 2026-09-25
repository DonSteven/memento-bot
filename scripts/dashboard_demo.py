"""Serve the real Dashboard UI with synthetic observations and inert services.

Example: .venv/bin/python scripts/dashboard_demo.py --data-dir /tmp/nanobot-ui-demo
The data directory must not exist. Stop the server with Ctrl-C and remove that
temporary directory when finished.
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web

from nanobot.agent.knowledge_retrieval import KnowledgeResult
from nanobot.api.dashboard import create_dashboard_app
from nanobot.observability.hook import preview
from nanobot.observability.store import ObservabilityStore


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


def seed(store: ObservabilityStore) -> None:
    now = datetime.now(timezone.utc)
    rows = (
        ("demo-complete-03", 8, "completed", "completed", 740),
        ("demo-error-02", 29, "error", "error", 1200),
        ("demo-complete-01", 68, "completed", "completed", 1850),
    )
    for run_id, minutes_ago, status, reason, duration in rows:
        store.start_run(run_id, "cli:synthetic-demo", "cli", "demo-model",
                        (now - timedelta(minutes=minutes_ago)).isoformat())
        store.add_event(run_id, "memory", {"core_count": 2, "retrieved_count": 1},
                        duration_ms=18)
        store.add_event(run_id, "model", {
            "usage": {"prompt_tokens": 120, "completion_tokens": 32},
            "usage_reported": True, "stop_reason": "tool_calls",
            "tool_call_count": 1,
            "content_preview": preview("Synthetic demo: checking a project note."),
        }, iteration=0, duration_ms=520)
        store.add_event(run_id, "tools", {"calls": [{
            "call_id": "demo-call-1", "name": "kb_search",
            "arguments": preview({"query": "synthetic project note", "access_token": "DEMO_SECRET_DO_NOT_PERSIST"}),
            "status": "ok" if status == "completed" else "error",
            "result_preview": preview("Synthetic demo evidence: one local note; token=DEMO_SECRET_DO_NOT_PERSIST"),
        }]}, iteration=0, duration_ms=260)
        store.add_event(run_id, "model", {
            "usage": {"prompt_tokens": 170, "completion_tokens": 58},
            "usage_reported": True, "stop_reason": reason, "tool_call_count": 0,
            "content_preview": preview("Synthetic demo answer: the note was retrieved."),
        }, iteration=1, duration_ms=430)
        store.add_usage(run_id, 290, 90, True)
        store.finish_run(run_id, status, reason, duration,
                         preview("Synthetic demo provider error", 1024)["text"]
                         if status == "error" else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="new directory under the system temporary directory")
    parser.add_argument("--port", type=int, default=18791)
    args = parser.parse_args()
    destination = args.data_dir.resolve()
    if not destination.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        parser.error("--data-dir must be under the system temporary directory")
    if destination.exists():
        parser.error(f"refusing to reuse existing directory: {destination}")
    destination.mkdir(parents=True)
    store = ObservabilityStore(destination / "dashboard.db")
    store.initialize()
    seed(store)
    agent = SimpleNamespace(memory_service=FakeMemory(), knowledge_retriever=FakeKnowledge())
    app = create_dashboard_app(agent, FakeCron(), store)
    print(f"Synthetic demo data at {destination}; open http://127.0.0.1:{args.port}/dashboard/")
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
