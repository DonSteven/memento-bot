---
name: memory
description: Structured personal memory with core injection and dynamic FTS recall.
always: true
---

# Memory

## Structure

- `memory/memory.db` — Source of truth for structured memories and history events.
- `memory/MEMORY.md` — Generated complete view of the six memory categories. In P1, manual edits are not imported.
- `memory/HISTORY.md` — Generated event-log view. It is not loaded into context or used as an edit source.

The `personal_profile`, `preferences`, and `constraints` categories are always loaded in full. `projects`, `daily_life`, and `plans_commitments` are selected by FTS for the current query.

## Search Past Events

Choose the search method based on file size:

- Small `memory/HISTORY.md`: use `read_file`, then search in-memory
- Large or long-lived `memory/HISTORY.md`: use the `exec` tool for targeted search

Examples:
- **Linux/macOS:** `grep -i "keyword" memory/HISTORY.md`
- **Windows:** `findstr /i "keyword" memory\HISTORY.md`
- **Cross-platform Python:** `python -c "from pathlib import Path; text = Path('memory/HISTORY.md').read_text(encoding='utf-8'); print('\n'.join([l for l in text.splitlines() if 'keyword' in l.lower()][-20:]))"`

Prefer targeted command-line search for large history files.

## Auto-consolidation

Old conversations are consolidated when the session grows too large or when `/new` starts a fresh session. Recent short conversations remain only in session history until one of those triggers runs. Consolidation commits the event and complete memory snapshot to SQLite, then regenerates both Markdown views.

Do not edit either Markdown view in P1: bidirectional `MEMORY.md` synchronization is planned for P2.
