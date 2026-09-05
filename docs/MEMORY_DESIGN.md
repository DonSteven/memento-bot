# Structured Memory Design

Personal memory separates canonical facts, archived conversation events and readable
workspace views. `MemoryService` coordinates extraction, revisioned SQLite writes,
view export and query preparation. `StructuredMemoryPipeline` only extracts facts;
`MemoryDatabase` owns persistence. Context builders consume a prepared
`MemoryContext` and perform no hidden database or model calls.

Core categories (`personal_profile`, `preferences`, `constraints`) are included in
full. Dynamic categories (`projects`, `daily_life`, `plans_commitments`) use bilingual FTS and vector recall followed by reciprocal-rank fusion.
Relevance gates, dynamic TopK and a token budget select whole records. Core items do not occupy dynamic TopK and each fact is injected once.

Canonical facts contain an ID, main class, explicit subclass and text. Distinct
facts may share a subclass. Extraction returns a complete snapshot: omitted facts
are removed and a valid empty snapshot clears the set. Invalid output cannot
commit. Events, facts, FTS and revision change in one transaction; view export
status is reported separately from database commit.

Extraction retains token-compression and `/new` triggers. A failed commit cannot
advance the archive position or clear pending conversation history. Recent
conversation can remain in session messages until an archive trigger occurs.

`MEMORY.md` supports complete, validated snapshot edits with stable IDs for
unchanged facts. Revision checks and pending publication records detect conflicts
and recover interrupted exports. `HISTORY.md` remains an exported view.
See [Markdown synchronization](MEMORY_P2_IMPLEMENTATION.md). See [semantic retrieval](MEMORY_P3_IMPLEMENTATION.md) for embedding configuration
and index consistency.

See [the runtime contract](MEMORY_P1_IMPLEMENTATION.md) and
[the implementation roadmap](MEMORY_KNOWLEDGE_IMPLEMENTATION_PLAN.md).
This document is an edited English treatment of the memory research plan,
updated to the current implementation contracts.
