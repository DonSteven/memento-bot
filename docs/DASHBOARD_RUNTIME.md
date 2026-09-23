# Dashboard runtime (P1–P4)

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

## Build and open the UI

Use Node 20.19 or newer within the Node 20 line. From the repository root:

```bash
cd dashboard
npm ci
npm run build
```

Start `nanobot gateway`, then open `http://127.0.0.1:18790/dashboard/`.
The Runs page supports a direct link such as
`http://127.0.0.1:18790/dashboard/#/runs/RUN_ID`. If the UI has not been built,
`/dashboard/` returns 503 while REST stays available. For local UI development,
run `npm run dev` in `dashboard/`; Vite proxies `/api/dashboard` to the gateway
on port 18790.

P4 includes Overview, Runs, and Memory. Refresh manually to pick up changes
from another process. Knowledge, Tasks, and live updates are planned for later
stages.

The Memory page reads a committed SQLite snapshot. Refreshing it does not
synchronize Markdown or call the embedding provider. It shows the revision,
counts for all six classes, and records. Retrieval debug runs only when Search
is clicked: it synchronizes `MEMORY.md`, reads core memories, then calls the
existing hybrid dynamic search. Synchronization may write memory and invoke
the embedding provider. The requested limit is capped by the service's
`dynamic_top_k`. Dynamic hits are recall candidates; later token budgeting
can trim what an Agent actually uses. RRF scores, ranks, sources, and vector
similarity are the service's values, not confidence percentages. A conflict or
invalid Markdown returns 409; embedding failure returns 502; unavailable
memory service returns 503.

```bash
curl http://127.0.0.1:18790/api/dashboard/overview
curl 'http://127.0.0.1:18790/api/dashboard/runs?limit=50'
curl http://127.0.0.1:18790/api/dashboard/runs/RUN_ID
curl http://127.0.0.1:18790/api/dashboard/memory
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
