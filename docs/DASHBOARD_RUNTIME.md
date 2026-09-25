# Dashboard runtime (V1)

**Status: Implemented.** This is the current operating guide. See the
[design record](DASHBOARD_V1_PLAN.md).

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

## Run the gateway with a fixed workspace

Use one config file and workspace consistently. From the repository root,
set the paths for the current user (or override them for another workspace).
The Dashboard is part of this gateway process; opening its URL alone does not
start the Agent, Memory, Knowledge, or Cron services.

```bash
NANOBOT_CONFIG_PATH="${NANOBOT_CONFIG_PATH:-$HOME/.nanobot/config.json}"
NANOBOT_WORKSPACE_PATH="${NANOBOT_WORKSPACE_PATH:-$HOME/.nanobot/workspace}"
.venv/bin/nanobot gateway \
  --config "$NANOBOT_CONFIG_PATH" \
  --workspace "$NANOBOT_WORKSPACE_PATH"
```

Open `http://127.0.0.1:18790/dashboard/`. Stop the foreground gateway with
Ctrl-C and run the same command to restart it. For a background process, send
SIGINT to its recorded PID and wait for it to exit before restarting. Check
that port 18790 has no listener before starting a second gateway. The gateway
can stay alive when the optional Dashboard listener fails, so verify the URL
and `/api/dashboard/overview` after each start.

An existing Memory database with a different schema is rejected at startup.
For valuable data, follow `spec/MEMORY_KNOWLEDGE_UPGRADE.md` and run
`scripts/upgrade_memory_knowledge.py` against a stopped workspace. To discard
an explicitly identified, disposable Memory set, stop all writers to that
workspace, then remove only `memory/memory.db` and any adjacent `-wal`, `-shm`,
or `-journal` sidecars, plus that same workspace's `memory/MEMORY.md` and
`memory/HISTORY.md`. Keep sessions, cron, observability, benchmarks, and any
other workspace files. Start the current `MemoryService` and run its Markdown
synchronization to create a valid six-class empty view; a zero-byte
`MEMORY.md` is invalid. Do not edit the schema version in SQLite.

Knowledge uses `<workspace>/knowledge/web_knowledge.db`. A newly initialized
database has schema 4. Its page stays empty until an explicit Knowledge
retrieval searches and ingests a source. HTTP 200 may carry `insufficient` or
an error business status; verify `status`, `online_attempted`, evidence and
`ingested_urls`. Repeat the same query to check local reuse, then restart with
the same workspace to check persistence. The Memory search can synchronize
Markdown and call the embedding provider; a real Agent conversation, `/new`
archive, Knowledge query, or scheduled task can call external services and
may incur charges. Enable external channels and heartbeat only within the
approved run scope.

## Build and open the UI

Use Node >=20.19.0 within the Node 20 line, matching `dashboard/package.json`.
From the repository root:

```bash
cd dashboard
npm ci
npm run build
```

The build writes `nanobot/api/dashboard_static`. Build before creating a
wheel/sdist or doing a non-editable source install; both package formats include
those built files. In an editable checkout, build before opening the UI.
`npm ci` does not use paid services. Docker builds the UI before its final
Python install. The generated directory and `node_modules` are not committed.

Start `nanobot gateway`, then open `http://127.0.0.1:18790/dashboard/`.
The Runs page supports a direct link such as
`http://127.0.0.1:18790/dashboard/#/runs/RUN_ID`. If the UI has not been built,
`/dashboard/` returns 503 while REST stays available. For local UI development,
run `npm run dev` in `dashboard/`; Vite proxies `/api/dashboard` to the gateway
on port 18790.

The UI includes Overview, Runs, Memory, Knowledge, and Tasks. One WebSocket
connection receives content-free invalidations for committed Agent run writes
and task changes in the gateway process. It refreshes only the relevant GET
views, and coalesces nearby events. When disconnected, the page keeps its
loaded data, shows a warning, and offers manual Refresh. Reconnection rereads
the current REST views. Changes from a separate CLI or API process are visible
after manual refresh; there is no cross-process event bus.

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

The Knowledge page runs retrieval only when **Run retrieval** is clicked. The
request passes `query`, `doc_limit`, and `evidence_limit` to the gateway's active
KnowledgeRetriever. The workflow may call the model, embedding, rerank, and
online search/fetch APIs; online supplementation may write fetched content to
the knowledge database. There is no automatic retry or query on page load,
refresh, or navigation. The response is the retriever's `to_dict()` result:
`sufficient`, `insufficient`, `retrieval_error`, `assessment_error`, and
`online_error` are business statuses returned with HTTP 200. Available evidence,
missing points, and online errors remain visible even when the status is an
error. `fetched_urls` records URLs for which a fetch was attempted, whereas
`ingested_urls` records successful ingestion. An unavailable retriever returns
503; an unhandled upstream failure returns 502.

The Tasks page reads the gateway's current `CronService`, including disabled
jobs and their saved run history. It can enable, disable, or delete existing
jobs. Enabling recalculates the next run; disabling prevents future scheduling
but does not cancel a task already running. Deleting removes the schedule and
its history after confirmation. Task changes are saved to the existing
`jobs.json`; there is no separate Dashboard task store. Use Refresh to pick up
changes made by another process. Changes by the gateway's cron tool and
scheduler update this page automatically.

The event endpoint is `/api/dashboard/events` (WebSocket, same origin). It
sends `run.started`, `run.updated`, `run.finished`, or `task.changed` with only
the relevant `run_id` or `job_id`. Slow subscribers are closed; clients read
current state again through REST. Memory and Knowledge search POST requests
are never replayed after an event or reconnect.

```bash
curl http://127.0.0.1:18790/api/dashboard/overview
curl 'http://127.0.0.1:18790/api/dashboard/runs?limit=50'
curl http://127.0.0.1:18790/api/dashboard/runs/RUN_ID
curl http://127.0.0.1:18790/api/dashboard/memory
curl http://127.0.0.1:18790/api/dashboard/tasks
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

## Observation privacy limits

Tool outputs are previews capped at 4 KiB UTF-8; arguments and errors have
smaller limits. New previews recursively sanitize recognized case, separator
and camel-case variants of `api_key`, `token`, `password`, `secret`,
`authorization`, `access_key`, `access_token`, `refresh_token`, `client_secret`,
`cookie`, `set_cookie` and `x_api_key`. JSON object/array strings are parsed
before sanitization. Text rules cover base64 data URLs (including embedded
ones), Bearer/Basic Authorization, Cookie and Set-Cookie lines, and explicit
recognized `key=value` or `key: value` assignments with ordinary quoted or
unquoted values. Sanitization precedes UTF-8 truncation; `truncated` describes
the sanitized text length, not redaction.

These rules protect future supported previews, including run error summaries.
They do not rewrite existing rows or guarantee removal of unknown secret
formats, personal prose, session identifiers, provider logs or other stored
data. Do not treat the database as sanitized public data. Agent/model/tool
inputs and returned content are unchanged.
Overview covers runs started in the previous 24 hours. Running runs count in
`runs` but not the success-rate denominator. Latency excludes runs without a
duration; empty denominators are `null`. Hourly buckets are UTC.

## Reproduce without credentials

```bash
.venv/bin/python -m pytest tests/dashboard nanobot/tests/p6/test_entry_integration.py -q
```

The CI Dashboard job runs `npm ci`, `npm run build`, then verifies wheel and
sdist assets and installed HTTP resources outside the source tree with
`scripts/check_dashboard_package.py`. These software checks need no model,
search or rerank credentials; dependency downloads may use package registries.
To inspect the actual React UI without credentials, build it first and run:

```bash
.venv/bin/python scripts/dashboard_demo.py --data-dir /tmp/nanobot-ui-demo
```

The directory must be new and under the system temporary directory. Open
`http://127.0.0.1:18791/dashboard/` and `#/runs/demo-complete-03`; stop with
Ctrl-C and remove the temporary directory. The fixture uses synthetic
observations and inert Memory, Knowledge and Cron dependencies, not the full
live gateway. Its token and duration values are illustrative.
