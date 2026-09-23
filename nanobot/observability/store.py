"""Small SQLite store for dashboard runs and events.

Connections are intentionally opened inside each call so callers can run every
operation in ``asyncio.to_thread`` without sharing connections across threads.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ObservabilityStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=3)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=3000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    model TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms REAL,
                    status TEXT NOT NULL,
                    stop_reason TEXT,
                    error TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    usage_reported INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC, run_id DESC);
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    kind TEXT NOT NULL,
                    iteration INTEGER,
                    started_at TEXT NOT NULL,
                    duration_ms REAL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, event_id);
            """)

    def start_run(self, run_id: str, session_key: str, channel: str, model: str,
                  started_at: str | None = None) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO runs(run_id,session_key,channel,model,started_at,status) "
                       "VALUES(?,?,?,?,?,'running')",
                       (run_id, session_key, channel, model, started_at or utc_now()))

    def finish_run(self, run_id: str, status: str, stop_reason: str | None,
                   duration_ms: float, error: str | None = None) -> None:
        with self._connect() as db:
            db.execute("UPDATE runs SET finished_at=?,duration_ms=?,status=?,stop_reason=?,error=? "
                       "WHERE run_id=?", (utc_now(), duration_ms, status, stop_reason, error, run_id))

    def add_event(self, run_id: str, kind: str, data: dict[str, Any],
                  iteration: int | None = None, duration_ms: float | None = None,
                  started_at: str | None = None) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO events(run_id,kind,iteration,started_at,duration_ms,data_json) "
                       "VALUES(?,?,?,?,?,?)", (run_id, kind, iteration, started_at or utc_now(),
                                                duration_ms, json.dumps(data, ensure_ascii=False)))

    def add_usage(self, run_id: str, prompt_tokens: int, completion_tokens: int,
                  reported: bool) -> None:
        with self._connect() as db:
            db.execute("UPDATE runs SET prompt_tokens=prompt_tokens+?, "
                       "completion_tokens=completion_tokens+?, "
                       "usage_reported=MAX(usage_reported,?) WHERE run_id=?",
                       (prompt_tokens, completion_tokens, int(reported), run_id))

    @staticmethod
    def _cursor_decode(cursor: str) -> tuple[str, str]:
        try:
            values = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
            if (not isinstance(values, list) or len(values) != 2
                    or not all(isinstance(value, str) for value in values)):
                raise ValueError
            return values[0], values[1]
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error) as exc:
            raise ValueError("Invalid cursor") from exc

    def list_runs(self, limit: int = 50, cursor: str | None = None) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        clause = ""
        params: list[Any] = []
        if cursor:
            started_at, run_id = self._cursor_decode(cursor)
            clause = "WHERE (started_at, run_id) < (?, ?)"
            params.extend((started_at, run_id))
        with self._connect() as db:
            rows = db.execute(f"SELECT * FROM runs {clause} ORDER BY started_at DESC, "
                              "run_id DESC LIMIT ?", (*params, limit + 1)).fetchall()
        items = [dict(row) for row in rows[:limit]]
        next_cursor = None
        if len(rows) > limit:
            last = items[-1]
            next_cursor = base64.urlsafe_b64encode(json.dumps(
                [last["started_at"], last["run_id"]]).encode()).decode()
        return {"items": items, "next_cursor": next_cursor}

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None
            events = db.execute("SELECT * FROM events WHERE run_id=? ORDER BY event_id",
                                (run_id,)).fetchall()
        return {"run": dict(row), "events": [
            {**dict(event), "data": json.loads(event["data_json"])} for event in events
        ]}

    def get_overview(self, now: datetime | None = None) -> dict[str, Any]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        since = now - timedelta(hours=24)
        with self._connect() as db:
            runs = [dict(row) for row in db.execute(
                "SELECT * FROM runs WHERE started_at>=? AND started_at<? ORDER BY started_at DESC,run_id DESC",
                (since.isoformat(), now.isoformat()))]
            errors = db.execute("SELECT data_json FROM events WHERE kind='tools' AND run_id IN "
                                "(SELECT run_id FROM runs WHERE started_at>=? AND started_at<?)",
                                (since.isoformat(), now.isoformat())).fetchall()
        terminal = [run for run in runs if run["status"] != "running"]
        durations = sorted(run["duration_ms"] for run in terminal if run["duration_ms"] is not None)
        buckets = []
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        for offset in range(23, -1, -1):
            start = hour_start - timedelta(hours=offset)
            end = start + timedelta(hours=1)
            sample = [run for run in runs if start <= datetime.fromisoformat(run["started_at"]) < end]
            sample_durations = [run["duration_ms"] for run in sample if run["duration_ms"] is not None]
            buckets.append({"hour": start.isoformat(), "runs": len(sample),
                            "avg_latency_ms": sum(sample_durations) / len(sample_durations)
                            if sample_durations else None})
        return {
            "window_start": since.isoformat(), "window_end": now.isoformat(),
            "runs": len(runs),
            "success_rate": sum(run["status"] == "completed" and
                                run["stop_reason"] == "completed" for run in terminal) / len(terminal)
                            if terminal else None,
            "avg_latency_ms": sum(durations) / len(durations) if durations else None,
            "p95_latency_ms": durations[math.ceil(.95 * len(durations)) - 1] if durations else None,
            "tool_errors": sum(call.get("status") == "error" for row in errors
                               for call in json.loads(row["data_json"]).get("calls", [])),
            "hours": buckets, "recent_runs": runs[:10],
        }


def new_run_id() -> str:
    return str(uuid.uuid4())
