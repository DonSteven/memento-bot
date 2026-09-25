# Dashboard V1 design record

**Status: Implemented.** This record explains the engineering choices behind
the current Dashboard. For commands and current contracts, use the
[runtime guide](DASHBOARD_RUNTIME.md). Dashboard V1 stage labels below
are distinct from [Memory/Knowledge P7](../spec/MEMORY_KNOWLEDGE_IMPLEMENTATION_PLAN.md#p7-evaluation-and-calibration),
which remains Planned.

## Problem and scope

The Agent already had Memory, Knowledge and Cron services, but local operators
could not easily inspect one turn's model/tool steps or diagnose those services
from a single UI. V1 provides five local pages: Overview, Runs, Memory,
Knowledge and Tasks. It is an operating console, not a general tracing platform
or production monitoring service. It shares the running gateway's services
instead of creating another Agent or retrieval backend.

## Architecture and event flow

1. `AgentLoop` creates a run before Memory preparation and ends it after the
   foreground session save. `ObservabilityHook` receives existing runner
   boundaries for model and tools; Memory and run-level errors are recorded by
   `AgentLoop` itself. Slash commands are excluded; system turns are included.
2. A separate SQLite database at `<workspace>/observability/dashboard.db`
   stores runs, aggregate usage and bounded event previews. Observation writes
   run in threads and are best effort so a store failure does not fail Agent
   work. Previews sanitize declared credential and data-URL formats before
   truncation. The [runtime guide](DASHBOARD_RUNTIME.md#observation-privacy-limits)
   states the exact privacy boundary.
3. The gateway serves REST under `/api/dashboard/*`, a content-free WebSocket
   invalidation stream at `/api/dashboard/events`, and built React assets under
   `/dashboard/`. The browser fetches current state through REST. It does not
   treat WebSocket messages as the source of record.
4. Memory, Knowledge and Tasks endpoints call the same `MemoryService`,
   `KnowledgeRetriever` and `CronService` held by the gateway. They preserve
   service result shapes and failure states instead of recomputing retrieval
   scores or duplicating scheduling rules.

Relevant code: [hook](../nanobot/observability/hook.py),
[store](../nanobot/observability/store.py),
[AgentLoop](../nanobot/agent/loop.py),
[REST/WebSocket](../nanobot/api/dashboard.py),
[React pages](../dashboard/src/pages/), and
[gateway lifecycle](../nanobot/cli/commands.py).

## Historical design decisions and current behavior

| Decision | Reason and implemented contract |
| --- | --- |
| Independent observation SQLite | Run history should remain available without changing the Memory or Knowledge schemas. It is workspace scoped and private. |
| Hook plus explicit loop boundaries | Hooks capture model/tools without replacing runner behavior; `AgentLoop` supplies Memory, terminal state and exceptions that hooks alone cannot see. |
| Bounded previews | Keep the trace useful without copying entire tool responses. Supported secrets and base64 data URLs are removed before byte truncation; unknown formats and historical rows remain outside the guarantee. |
| Read-only Memory GET | A page refresh must not silently synchronize Markdown or call an embedding provider. Manual Search can do both. |
| Explicit Knowledge action | Retrieval may use paid model, embedding, rerank, search and fetch APIs. The page does not invoke it on mount, refresh or reconnect. |
| Shared CronService | Enable, disable and delete operate on the existing scheduler and `jobs.json`, with the same task semantics. |
| Same-process invalidations | A single lightweight WebSocket updates gateway writes. Writes by separate CLI/API processes require manual refresh. |
| Loopback default | The UI has no login; `127.0.0.1` is the default listener. Remote exposure and authentication are outside V1. |

Runs can end `completed`, `stopped`, `error` or `cancelled`. Hard process exit
may leave `running`. Model duration includes request preparation/retries inside
the hook boundary, while tools duration covers a batch. Usage is the sum of
reported model counts; nested Knowledge/embedding/background usage is excluded.
Overview's 24-hour window is based on run start times, with UTC hourly buckets.

The frontend uses a single WebSocket connection and coalesces invalidations.
After disconnection it keeps loaded data and offers Refresh; reconnect rereads
REST. No Memory or Knowledge POST is replayed. Static files are built with
TypeScript/Vite and included in wheel and sdist artifacts after the build.

## Failure and cancellation boundaries

Observation failures are logged but do not replace Agent output. Tool/model
responses and caller-owned dictionaries are not mutated for tracing. A
cancelled run receives a terminal state when the process can finish the write;
a killed process cannot promise that write. Unavailable stores return 503 from
the REST API. Memory validation conflicts return 409, upstream failures 502
and unavailable services 503. Knowledge can return HTTP 200 with structured
`insufficient` or error status, preserving any partial evidence.

## Verification and limits

Automated contracts live in [Dashboard tests](../tests/dashboard/) and the
[entry integration tests](../nanobot/tests/p6/test_entry_integration.py).
Run them with the command in the [runtime guide](DASHBOARD_RUNTIME.md#reproduce-without-credentials).
The CI workflow also builds the React UI, checks generated resources in
wheel/sdist, then serves both installed package forms outside the checkout.
Memory and Knowledge P7 evaluation remains Planned. V1 has no cross-process event bus,
authentication, production observability guarantee or automatic cleanup of
already persisted sensitive rows.
