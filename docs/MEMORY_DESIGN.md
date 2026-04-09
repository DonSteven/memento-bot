# Structured Memory Design

## Purpose

Store durable personal facts independently of the Markdown text used to display
them. The design separates conversation records, the current set of canonical
facts, and readable workspace views. It preserves the existing token-driven
consolidation boundary and append-only session message history.

This document is an edited English version of the memory research plan, scoped
to the structured SQLite and FTS implementation described here. Field names and
interfaces follow the implementation rather than the research pseudocode.

## Data flow and ownership

1. `MemoryConsolidator` selects a conversation chunk at a user-turn boundary.
2. `MemoryStore` routes v2 consolidation to `StructuredMemoryPipeline`.
3. The pipeline reads the current canonical snapshot from SQLite and asks the
   model for a history entry and a complete replacement snapshot.
4. The database stores the event and replaces the canonical snapshot in one
   transaction, rebuilding its FTS index.
5. The database exports `memory/MEMORY.md` and `memory/HISTORY.md`.

The consolidation cursor advances only after consolidation reports success.
Session messages remain append-only. The database is the source for subsequent
v2 consolidation prompts; editing the exported Markdown does not import those
edits into the canonical snapshot.

## Storage layers

| Layer | Contents | Role |
| --- | --- | --- |
| `raw_events` | Timestamp, session, history summary, source text and source range metadata | Retain an inspectable record of archived conversations |
| `canonical_memories` | Memory ID, main class, subclass and text | Represent the current durable facts |
| `canonical_fts` | Searchable canonical text and subclass | Retrieve facts relevant to the current query |
| Markdown views | Formatted canonical facts and chronological history | Provide readable workspace files |

The database lives at `workspace/memory/memory.db`. Schema versions describe the
supported storage shape; unsupported layouts are rejected explicitly. The
database module owns SQL and rendering. The pipeline owns model requests,
payload validation and extraction failure handling.

Canonical items are a better indexing unit than filenames: a filename provides
little information about an individual fact. They also avoid tying search
identity to Markdown section positions that can change when a view is rendered.

## Categories and identity

The fixed main classes are:

- `personal_profile`: stable background information.
- `preferences`: communication and collaboration preferences.
- `constraints`: rules and boundaries that must be respected.
- `projects`: ongoing learning and work.
- `daily_life`: routines, interests and hobbies.
- `plans_commitments`: future plans, deadlines and commitments.

Each fact must have an explicit, non-empty subclass and text. A subclass groups
related facts; it is not a unique slot. Several distinct facts can share the
same subclass. Identity incorporates the main class, subclass and text so those
facts survive together, while repeated identical facts can be deduplicated.

The extraction result is a complete current snapshot. The model must retain
still-valid facts and omit obsolete facts. An empty valid snapshot clears the
canonical set. Malformed output is not interpreted as permission to clear it.

## Writing and failure handling

The model receives the current database-rendered snapshot and the conversation
chunk. Its structured tool response supplies `history_entry` and
`canonical_memories`. Validation checks the response shape, category and
required fields before persistence.

Event insertion and canonical replacement are transactional, so a failed write
does not leave half of a new snapshot committed. Index rebuilding belongs to
the same database operation. Markdown export follows database persistence;
these files are separate filesystem outputs, not part of SQLite's transaction.

The pipeline supports providers that reject a forced tool choice by retrying
with automatic tool selection. Repeated extraction failures eventually archive
the raw conversation without replacing the existing canonical snapshot. This
preserves conversation information while keeping invalid extraction output out
of the durable fact set.

## Context retrieval

Core categories (`personal_profile`, `preferences`, and `constraints`) are
included independently of lexical matches. FTS retrieves additional relevant
facts from the current canonical snapshot. Deduplication prevents core facts
from appearing a second time in the retrieved block.

Query preparation extracts usable lexical terms and builds a tolerant prefix
query. Results use SQLite FTS5 ranking with deterministic secondary ordering.
Blank queries and absent databases have explicit behavior covered by tests.

The research plan also explored canonical-item vector indexing and semantic
reranking. Those are separate retrieval stages; this document's implemented
memory retrieval contract is the SQLite FTS path.

## Rendering

`MEMORY.md` follows the six-category order and groups facts by subclass. Empty
sections keep their descriptive placeholders, including when input contains
only blank text. Stable ordering makes replay comparisons meaningful.

`HISTORY.md` presents archived events chronologically and remains useful for
text search. Neither exported view should be committed as a user's runtime
data. The template under `nanobot/templates/memory/` is distributable product
content.

## Verification

Use temporary workspaces and scripted model responses to check:

- Schema initialization and rejection of unsupported column layouts.
- Transactional event and snapshot persistence.
- Replacement, removal, empty snapshots and distinct facts sharing a subclass.
- Deterministic rendering and FTS results.
- Core-memory inclusion and retrieval deduplication.
- Database-derived consolidation prompts, including stale Markdown files.
- Invalid responses, provider tool-choice errors and raw archival.

Offline replay compares behavior without invoking a real model. Extraction
evaluation measures predicted facts against expected snapshots, including
preservation and removal. A synthetic replay demonstrates contract behavior;
it does not establish real-model quality or long-horizon benchmark performance.

The original research plan proposed longer synthetic conversations containing
stable preferences, changing project state and irrelevant dialogue. Such
experiments should record their inputs and methods separately from unit tests.
Evaluation datasets and generated reports remain outside the repository.
