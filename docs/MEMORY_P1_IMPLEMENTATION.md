# P1 Structured Memory Runtime

P1 replaces the former runtime modes with one SQLite-backed structured memory path.

## Runtime flow

1. `MemoryService` initializes `memory/memory.db` for a workspace.
2. Before preflight compression, `AgentLoop` awaits `MemoryService.prepare_context(query, retrieval_budget)`. Token estimation, message assembly, and the post-turn compression check reuse that prepared context.
3. The service reads all `personal_profile`, `preferences`, and `constraints` records and retrieves only `projects`, `daily_life`, and `plans_commitments` records through FTS.
4. `ContextBuilder` renders the prepared `MemoryContext`. The synchronous token estimator accepts that context and does not query memory or call the extraction model. `/status` explicitly prepares an empty-query context before estimating.
5. Context compression and `/new` invoke `MemoryConsolidator`, which asks `StructuredMemoryPipeline` for a complete new snapshot.
6. `MemoryDatabase.commit_snapshot` commits the history event, full snapshot, FTS rows, and revision in one transaction. The consolidation checkpoint advances only after that commit.
7. After a committed compression archive, the consolidator refreshes context using the actual query and returns it for subsequent estimation and message assembly. Failed or skipped archives do not trigger a refresh.
8. `MEMORY.md` and `HISTORY.md` are generated views. A failed view export is reported separately and does not cause the committed conversation to be extracted again.

## Configuration and current limits

There is no `memory.mode` setting. Supplying it is a configuration error. All CLI, gateway, OpenAI-compatible serve, and SDK entry points use the structured path.

P1 does not import manual edits from `MEMORY.md`, perform vector retrieval, or change Chinese FTS tokenization. Those capabilities remain assigned to later phases.

Memory extraction is triggered only by token-based context compression and `/new`; it is not run after every conversational turn.

## Validation

Run the P1 and directly affected tests:

```bash
uv run --extra dev pytest -q nanobot/tests/p1 tests/agent/test_consolidate_offset.py tests/agent/test_loop_consolidation_tokens.py tests/agent/test_task_cancel.py tests/cli/test_restart_command.py nanobot/tests/phase_1/agent/test_memory_render.py nanobot/tests/phase_6/agent/test_memory_eval.py nanobot/tests/phase_6/agent/test_memory_extraction_eval.py nanobot/tests/phase_7/agent/test_memory_semantic_eval.py nanobot/tests/phase_8/agent/test_memory_query_eval.py
```

The default pytest and CI collection includes both `tests/` and `nanobot/tests/`.

Tests use scripted model responses and SQLite FTS. Coverage includes prepared-context
reuse, refresh after committed archival, invalid extraction leaving storage unchanged,
snapshot replacement and clearing, schema validation, and lexical variants.
The fixed lexical retrieval sample is a deterministic fixture, not a quality benchmark.
