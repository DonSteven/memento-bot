# P2 Markdown Memory Synchronization

P2 makes `memory/MEMORY.md` a supported editing surface for the SQLite-backed memory snapshot. `HISTORY.md` remains an exported history view and is never imported.

## Editing format

`MEMORY.md` must remain a complete document with these six headings, in this order:

1. `Personal Profile`
2. `Preferences`
3. `Constraints`
4. `Projects`
5. `Daily Life`
6. `Plans and Commitments`

Each memory uses one line:

```markdown
- sub_class: Memory text
```

Subclasses and memory text must be non-empty single lines. Subclasses cannot contain the delimiter `: `; colons in the memory text are allowed. Extraction normalizes surrounding whitespace, then applies the same field validation used by database writes and rendering. Unrepresentable model output is rejected before committing a snapshot or exporting a view.

Example:

```markdown
## Preferences

- response_style: Prefer concise answers
- language: Prefer Chinese for project discussions
```

Keep the generated placeholder in a category with no entries. A complete generated template containing only placeholders is a valid request to clear the memory snapshot. A missing file is rebuilt from SQLite; a zero-byte file is rejected and never treated as a clear request.

Unknown or repeated headings, reordered or missing categories, malformed entries, empty subclasses, empty text, duplicate entries, and arbitrary text outside the document structure are rejected. The error reports the first invalid line. Validation is all-or-nothing: an invalid file is preserved and no database rows or revision are changed.

## When edits take effect

Manual edits are checked before context preparation, model extraction, and raw history archival. A valid edit is imported transactionally before memory is read, so the next request sees updated core memory and dynamic FTS results. Re-reading an unchanged file does not add records or increment the revision.

Unchanged queries do not read or rewrite the full history. History is exported on publication or recovery, and rebuilt if the file is missing. Evaluation fixtures are parsed and rendered into canonical Markdown before their database synchronization baseline is established; legal whitespace differences do not cause a conflict.

Chat channels report memory validation and conflict messages directly, including the first invalid line when available. The HTTP endpoint returns status `409` with error type `memory_sync_error` and the same reason; it does not call the model or retry the request for these errors.

Changes are compared by `(category, sub_class, text)`. Exact unchanged entries keep their existing IDs. Editing text or a subclass, or moving an entry between categories, is persisted as removal of the old entry plus addition of the new entry. Multiple different facts may share a subclass.

## Conflict and recovery behavior

All in-process writes use the same `MemoryService` lock. After a potentially slow extraction returns, the service checks the database revision and the exact `MEMORY.md` text again. If either changed, the stale extraction result is discarded; this state is not retried as an extraction failure and does not trigger raw archival.

Every database write records a pending publication containing:

- the committed revision;
- the exact file text expected before replacement;
- the exact normalized target text.

The view is written to a temporary file and atomically replaces `MEMORY.md`. `HISTORY.md` is also published before the database marks that publication complete. A history export failure therefore retains the pending state, even if `MEMORY.md` is already updated. On startup and before reads or writes:

- if the file is still the expected old text, the pending target is published;
- if the file already equals the target, history is exported and publication is confirmed;
- if it equals neither, a conflict is reported and both the human file and database target are preserved.

No automatic merge or winner selection is performed. Resolve a conflict by backing up both versions, choosing the desired complete `MEMORY.md`, and restoring consistency before retrying. External editors are not controlled by the service lock. Changes detected by the check immediately before replacement are preserved; the check and replacement do not provide a cross-process compare-and-swap guarantee.

## Validation

Run the P2 contract and directly affected regression tests:

```bash
uv run --extra dev pytest -q nanobot/tests/p2 nanobot/tests/p1 nanobot/tests/phase_1/agent/test_memory_render.py tests/agent/test_consolidate_offset.py tests/agent/test_loop_consolidation_tokens.py tests/agent/test_task_cancel.py tests/cli/test_restart_command.py nanobot/tests/phase_6/agent/test_memory_eval.py nanobot/tests/phase_6/agent/test_memory_extraction_eval.py nanobot/tests/phase_7/agent/test_memory_semantic_eval.py nanobot/tests/phase_8/agent/test_memory_query_eval.py
uv run --extra dev pytest -q tests/test_openai_api.py
```

Tests use temporary workspaces, SQLite FTS and scripted model responses. Coverage includes edit import, strict parsing, conflict detection, interrupted publication recovery and API error propagation. No live model backend is required.
