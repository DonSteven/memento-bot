# Dashboard Screenshots and Animated Demo

Status: **Complete for local delivery** (2026-09-25). GitHub-render verification is a separate post-publication check.

## Objective and completion state

Produce a clear README demonstration using the real Dashboard UI and explicitly synthetic data:

1. An Overview screenshot showing run totals, charts and recent runs.
2. A focused Runs detail screenshot showing Memory, both Model steps, expanded Tools, status and usage.
3. A short GIF showing a new run appearing, live updates, opening its detail and expanding a tool result.

At completion, all three assets are embedded in the README, a documented local command can regenerate them, and the capture process requires no account credentials or paid services. This is a presentation enhancement to the completed public release readiness phase, not a rerun of live-provider acceptance.

## Baseline and scope

- `scripts/dashboard_demo.py` already serves the real Dashboard against temporary synthetic observations and inert Memory/Knowledge/Cron dependencies.
- `docs/assets/dashboard/overview.png` and `runs.png` already exist and are referenced in the README. Inspect and reuse their composition where suitable; refresh them only to meet the unified capture contract.
- The existing fixture seeds completed runs but does not replay a live run with corresponding same-process WebSocket events.
- Browser launch and PNG capture have been verified with an explicitly selected cached Chromium executable. The installed Playwright package's default browser revision did not match the cache.
- Linux Node 20.19 and Playwright are available in temporary tooling locations. These machine-specific paths must not be committed as defaults.
- Playwright's cached FFmpeg supports PNG and VP8 but lacks a GIF encoder. GIF encoding requires an additional local development tool.

Reuse `ObservabilityStore`, `DashboardEvents`, `create_dashboard_app` and existing React views. Add only demo/capture helpers, focused tests, assets and documentation. No production API, database schema, product UI feature, hosted demo, real gateway restart or private workspace access is required.

## Capture and data contract

### Genuine UI, synthetic execution

The replay writes synthetic records through the current observation store methods and publishes `run.started`, `run.updated` and `run.finished` only after the associated writes succeed. React receives the existing WebSocket invalidations and fetches real local REST responses.

Do not change displayed text with DOM injection, substitute prerecorded REST responses, or draw imitation application screens. This demonstrates the UI/REST/store/event integration; it does not claim to run a real model, actual knowledge retrieval or the complete AgentLoop.

Use a small, fixed set of readable English project examples. Keep model names explicitly demonstrative, exclude credentials and personal identifiers, and keep counts/usage consistent with the seeded events. Synthetic durations are illustrative; deliberate presentation pauses are not measured service latency.

### Isolation and control

- Create a new temporary demo directory for each generation. Never load default user config or use `~/.nanobot/workspace`.
- Bind the demo listener to loopback on an available port and report the actual URL to the capture process.
- Keep Memory/Knowledge dependencies inert; no click should reach model, embedding, rerank or search services.
- Drive replay steps over the demo child process's stdin with a small fixed command set and stdout acknowledgments. Reserve stdout for control messages in capture mode and send logs to stderr. Do not add an HTTP replay endpoint or a generalized orchestration protocol.
- Use explicit start/update/finish acknowledgments plus browser DOM assertions to coordinate recording. A short visual hold may follow an assertion; elapsed time alone is not proof that a state rendered.
- On exit, close the browser, demo listener and encoder process and remove temporary stores/frames. Preserve any failed capture diagnostics only in a clearly reported temporary directory.

### Asset targets

| Asset | Composition | Output target |
| --- | --- | --- |
| `overview.png` | Overview heading, summary cards, nonempty hourly chart and recent runs | 1440 × 1000 CSS-pixel viewport, device scale 1; target ≤1 MiB |
| `runs.png` | Focused real run detail with Memory, completed status, usage, expanded tool arguments/result and final Model answer | Capture the detail panel at device scale 1; target ≤1 MiB |
| `demo.gif` | One continuous run scenario | 15–25 seconds, 6–10 fps, target ≤5 MiB |

Use normal application styling. Fix locale to English and timezone to UTC. Seed timestamps relative to one captured base time so the 24-hour Overview remains populated on later reruns. Identical output bytes across dates are not required.

Start recording at the Overview PNG viewport size; encode the GIF at 1120–1280 pixels wide if needed. Prefer fewer frames or shorter idle holds before reducing text readability. Size limits are targets: if the final GIF remains above 5 MiB after reasonable optimization, report the measured size and tradeoff instead of silently replacing it with a poor-quality clip or a different format.

Every README asset has a “Synthetic demo data” caption and English alternative text. The GIF also carries a small, unobtrusive “Synthetic demo data” label outside the application content, added during encoding, so it remains labeled when shared separately. Do not cover or alter application state.

## Storyboard

| Approximate time | Visible action | Required evidence |
| --- | --- | --- |
| 0–3 s | Runs list displays baseline fixture data | Page ready; WebSocket connected |
| 3–6 s | New synthetic run appears as running | `run.started` received and matching list row rendered without manual refresh |
| 6–10 s | Open the new run; Memory and first Model events arrive | Detail has the same run ID and actual stored events |
| 10–16 s | Tool event arrives; expand `kb_search` | Arguments and synthetic result are legible |
| 16–21 s | Final Model event and completed status appear | Stored terminal state and UI both show completed; usage totals agree |
| 21–23 s | Hold final frame | Viewer has time to read before the loop restarts |

Timings guide presentation, not test synchronization. Stay on Runs for the GIF; the Overview PNG supplies the overview. Do not manufacture intermediate statuses that the product does not support.

## Ordered tasks

### T01 — Verify source and local capture prerequisites

- [x] Inspect Git status and preserve unrelated changes.
- [x] Inspect current screenshots, fixture helper and built frontend assets.
- [x] Resolve working Node, Playwright and Chromium; verify a temporary browser screenshot. Support documented executable overrides instead of committed `/tmp` or personal paths.
- [x] Prepare a separate development environment for the capture helper. Reuse repository/tooling conventions when possible; do not add Playwright or encoding tools to production Python dependencies.
- [x] Obtain a full FFmpeg development binary if needed; verify GIF encoding and palette generation with a tiny temporary sample. Record installation and version-check instructions.

Acceptance: screenshot and GIF encoding prerequisites actually work; no paid service calls or user configuration reads occur.

### T02 — Extend the demo helper with controlled live replay

Target: `scripts/dashboard_demo.py`.

- [x] Preserve its existing standalone static-demo command.
- [x] Share one `DashboardEvents` instance between the demo app and replay driver.
- [x] Add explicit capture mode with stdin steps for starting a new run, adding Memory/Model/Tools events and finishing it; keep control logic local to the script.
- [x] Write using current store methods and publish matching invalidations after persistence.
- [x] Report listener readiness, selected port, fixture run IDs and completed steps to the parent process.
- [x] Implement clean shutdown on EOF or explicit stop; return a clear error for an unknown/out-of-order step rather than silently generating inconsistent data.

Acceptance: a controlled run advances from absent to running to completed with consistent events/usage, using the same app/store/event instance.

### T03 — Validate replay semantics with focused tests

Target: a small `tests/dashboard/test_demo_replay.py` or an equivalent existing test location.

- [x] Exercise the finite replay with a temporary store and an actual event subscriber.
- [x] Assert notifications observe already-persisted state, IDs match, usage is consistent, and completion occurs once.
- [x] Verify invalid control input fails clearly and temporary demo setup refuses an existing populated directory.
- [x] Reuse existing Dashboard API/WebSocket tests for general transport behavior; do not duplicate the full test suite.

Acceptance: focused tests pass without network credentials. No tests are needed merely to assert screenshot filenames or prose wording.

### T04 — Implement repeatable browser capture

Target: `scripts/capture_dashboard_demo.mjs`, with its development dependency manifest/lockfile isolated from production dependencies if required.

- [x] Expose one documented capture command, output directory and optional browser/encoder executable overrides.
- [x] Spawn the Python demo helper in capture mode and wait for its readiness acknowledgment.
- [x] Launch Chromium with the fixed viewport/locale/timezone; wait for fonts, data and animations to settle.
- [x] Capture Overview and a focused completed run detail with its first Model step collapsed, tool result expanded and final Model answer visible, using accessible text/role selectors.
- [x] Return to the Runs list, start frame capture, and execute the storyboard through replay commands and real UI clicks.
- [x] Assert new-row appearance, selected ID, added events, expanded result and completed state. Collect browser console errors, failed local requests and unexpected nonlocal requests.
- [x] Capture PNG frames at a bounded cadence for GIF encoding; use timeouts for readiness failures and always clean up child processes in `finally`.

Acceptance: the helper generates both screenshots and a complete frame sequence without human clicking; observed changes arrive via WebSocket/REST, not reloads or DOM edits.

### T05 — Encode and inspect the short GIF

Dependencies: T04.

- [x] Encode the sequence using FFmpeg palette generation/use, loop settings and the synthetic-data label.
- [x] Measure dimensions, duration, frame count and file size; remove duplicate idle frames where effective.
- [x] Inspect beginning, new-row appearance, details, tool expansion, terminal state and loop boundary.
- [x] Check text at README display width, cursor/action clarity, palette artifacts and clipping. Adjust fixture text or presentation timing rather than altering production components for recording.
- [x] Save final `overview.png`, `runs.png` and `demo.gif` under `docs/assets/dashboard/`; do not commit raw frames, temporary databases or encoder binaries.

Acceptance: all specified stages are visible; PNG details are readable; GIF meets the intended duration and has a reported size. No real account or session data appears.

### T06 — Integrate README and reproduction instructions

Targets: `README.md`, `docs/DASHBOARD_RUNTIME.md`.

- [x] Place the Demo after the short README introduction and three contribution highlights; retain the two stills with full-size links, avoiding duplicate copies of the assets.
- [x] Add clear English captions and alternative text. Explain that the UI is real and execution data/timing is synthetic.
- [x] Document dependency setup and the single regeneration command, expected outputs and optional executable overrides.
- [x] Explain the demo's scope: local observation playback, no actual LLM/web search and no account setup.
- [x] Link this plan from the demo instructions; leave the completed release-readiness phase's historical status intact.

Acceptance: local relative links resolve, stills remain available for readers who cannot or prefer not to view animation, and no GitHub upload/hosting dependency is required for the assets.

### T07 — Final local validation and handoff

- [x] Regenerate once into a fresh temporary workspace using the documented command to prove reproducibility, not just manual one-off capture.
- [x] Run affected demo/event tests; build the frontend only if required by source/build changes.
- [x] Verify the three final assets, README rendering, captions and absence of browser/request errors.
- [x] Confirm no demo process remains and no private runtime files were changed.
- [x] Record actual commands, tool versions, asset sizes/dimensions/duration and test outcomes below.
- [x] Mark this plan Complete for local delivery only after required checks pass. Keep actual GitHub-render verification explicitly pending until the changes are published and inspected.

Acceptance: reviewable local assets and reproducible helpers are delivered. Stop before commit/push/publication unless separately authorized.

## Acceptance checklist

| ID | Required result | Evidence | Status |
| --- | --- | --- | --- |
| A01 | Real Dashboard with isolated synthetic data and no paid calls | Demo setup and capture network log | Complete |
| A02 | New run and updates reach React through real store/REST/WebSocket | Replay tests and browser state assertions | Complete |
| A03 | Overview PNG shows useful summary, chart and recent runs | Visual inspection, dimensions/size | Complete |
| A04 | Runs PNG shows Memory, Model, Tools and expanded tool details | Visual inspection, dimensions/size | Complete |
| A05 | GIF visibly follows the storyboard and ends completed | Playback review, 15–25 s duration, measured size | Complete |
| A06 | All assets are clearly labeled synthetic and contain no private data | Asset/caption review | Complete |
| A07 | One documented command regenerates all outputs in a fresh workspace | Successful second generation | Complete |
| A08 | README references work and capture resources are cleaned up | Local rendering, link and process checks | Complete |

All A01–A08 are required. File-size exceptions must be disclosed with a legibility-preserving option; do not call an unreadable GIF acceptable solely because it meets a byte budget.

## Local delivery evidence (2026-09-25)

- Python 3.13.12; Node 20.19.0; Playwright 1.63.0; Chrome for Testing 149.0.7827.55; FFmpeg 7.0.2-static. The browser and encoder paths were passed as overrides and are not repository defaults.
- Verified a temporary Chromium screenshot and a temporary FFmpeg palettegen/paletteuse GIF sample. Installed Playwright in the isolated `scripts/dashboard-capture` development directory and obtained the full FFmpeg binary through `imageio-ffmpeg==0.6.0` in `/tmp`.
- Ran `.venv/bin/python -m pytest tests/dashboard/test_demo_replay.py tests/dashboard/test_api_events.py -q`: 6 passed. The replay tests check persisted state when notifications arrive, event IDs/counts, usage totals, completion once, invalid order, directory refusal, and that every synthetic run duration covers its sequential event durations.
- Ran `node scripts/capture_dashboard_demo.mjs --output-dir /tmp/nanobot-dashboard-revision --browser <cached-chrome> --ffmpeg <full-ffmpeg>` in a fresh temporary workspace. The full capture passed browser DOM, WebSocket, REST and request checks with no browser/request errors; its verified assets were copied into `docs/assets/dashboard/`.
- Final assets: `overview.png` 1440×1000, 93,664 bytes; `runs.png` 878×969, 64,455 bytes; `demo.gif` 1200×868, 367,178 bytes, 164 frames at 8 fps (about 20.5 seconds). Reviewed baseline, new row, Memory/Model, expanded Tools result, final Model and completed state; the external synthetic-data label is visible. The README places Demo before detailed contributions and provides full-size links for both stills.
- Confirmed README asset links resolve, `git diff --check` is clean, and the capture leaves no demo server or encoder running. No private runtime files were changed. GitHub rendering was outside the local acceptance check.

## User involvement and publication boundary

The agent handles local fixture changes, temporary tool setup, capture automation, encoding, tests and README integration. User operation of the browser or encoder is not required.

After local delivery, the user may request presentation changes such as pacing or emphasis. Git commit/push and public publication require separate direction. Account login or repository permission issues, if encountered at that later stage, may require user action; they do not block local asset creation.

No GitHub rendering success is claimed from a local preview alone. No real model execution or retrieval-quality result is claimed from this synthetic demonstration.
