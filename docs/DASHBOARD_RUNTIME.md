# Dashboard runtime (P1–P2)

The gateway starts a local REST listener by default at `127.0.0.1:18790`.
It uses the gateway's existing `AgentLoop` and `CronService`. The independent
`nanobot serve` command does not host this listener, but its Agent runs write to
the same workspace-scoped SQLite file.

```json
{
  "dashboard": {"enabled": true, "host": "127.0.0.1", "port": 18790}
}
```

The store is `<workspace>/observability/dashboard.db`. If port binding or store
initialization fails, the gateway continues. A request whose store is unavailable
returns HTTP 503. Use a different Dashboard port for each gateway process on
the same host. The listener has no login and is intended for local use.

```bash
curl http://127.0.0.1:18790/api/dashboard/overview
curl 'http://127.0.0.1:18790/api/dashboard/runs?limit=50'
curl http://127.0.0.1:18790/api/dashboard/runs/RUN_ID
```

The runs endpoint returns `items` and `next_cursor`. Pass the cursor unchanged
in the next request. Unknown run IDs return 404; invalid pagination returns 400.
All stored timestamps are UTC ISO 8601. `duration_ms` uses a monotonic clock.

One non-command Agent execution creates one run. System messages are included;
subagent internals and slash commands are not. A run ends after the foreground
session save. Status is `completed`, `stopped` (iteration/context limit),
`error`, or `cancelled`. A process kill or unusable disk can leave `running`.
Provider usage is summed per iteration only when reported and excludes nested
knowledge/embedding/background calls. `usage_reported` distinguishes missing
usage from reported zero. The model duration is the runner hook boundary,
including request preparation/retries/stream callbacks; tools duration is for
the batch, not each tool call.

Tool outputs are previews capped at 4 KiB UTF-8; arguments and errors have
smaller limits. Known credential keys are redacted, but arbitrary personal text
can still be present. Do not treat the database as sanitized public data.
Overview covers runs started in the previous 24 hours. Running runs count in
`runs` but not the success-rate denominator. Latency excludes runs without a
duration; empty denominators are `null`. Hourly buckets are UTC.
