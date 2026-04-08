# Memory Phase 6

Phase 6 closes the loop around the staged memory refactor by adding a repeatable offline replay runner, a fixed-format ablation report, and a lightweight operator workflow for debugging and rollback.

## What Phase 6 Adds

- `nanobot.agent.memory_eval`: fixture-driven replay runner for `legacy` and `v2`
- `nanobot memory-eval`: CLI entrypoint that prints the report and can save JSON and Markdown copies
- Fixed-format mode summary:
  - `passed`
  - `failed`
  - `not_implemented`
- Explicit `not_implemented` reporting for `vec` and `hybrid` until Phase 5 lands

## How To Run

From the repository root:

```bash
nanobot memory-eval
```

Print JSON instead of Markdown:

```bash
nanobot memory-eval --output json
```

Save both report formats:

```bash
nanobot memory-eval --save-dir ./artifacts/memory-eval
```

If you want to point at a different checkout that still contains `nanobot/tests/phase_*` fixtures:

```bash
nanobot memory-eval --fixtures-root /path/to/repo
```

## Report Coverage

The Phase 6 replay suite currently covers:

- `boundary_replay`
- `legacy_consolidation_replay`
- `v2_persistence_replay`
- `v2_retrieval_replay`

The suite is deliberately offline and fixture-driven. It does not call a live LLM provider.

## Debug Workflow

1. Run `nanobot memory-eval --output json --save-dir <dir>`.
2. Inspect the top-level `summary` and `mode_summary`.
3. If a scenario failed, open the saved JSON report and inspect:
   - `results[].scenario`
   - `results[].checks`
   - `results[].details`
4. Re-run the relevant phase test directly:
   - `nanobot/tests/phase_0/`
   - `nanobot/tests/phase_2/`
   - `nanobot/tests/phase_3/`
   - `nanobot/tests/phase_4/`
5. Only after the targeted phase passes again should you trust a regenerated Phase 6 report.

## Rollback Guidance

If the DB-backed path is unstable in production-like use, roll back in the config first instead of patching behavior ad hoc.

Conservative rollback:

```json
{
  "memory": {
    "mode": "legacy"
  }
}
```

Interpretation:

- `legacy`: only file-backed `MEMORY.md` / `HISTORY.md`
- `v2`: structured memory consolidation writes `raw_events + canonical_memories`, then re-renders compatible `MEMORY.md` / `HISTORY.md` and injects prompt memory via core memory plus FTS retrieval

`vec` and `hybrid` should not be configured yet because they are not implemented in the current codebase.
