# Memory and Knowledge Implementation Roadmap

This roadmap defines the structured memory and external knowledge work in seven
stages. Each stage owns its implementation, affected callers, regression tests
and documentation. A stage marked complete describes implemented contracts;
real-model quality and parameter calibration require separate evaluation.

## Scope and final behavior

| Area | Contract |
| --- | --- |
| Core memory | Include every `personal_profile`, `preferences` and `constraints` item once, including for empty or unrelated queries. |
| Dynamic memory | Retrieve `projects`, `daily_life` and `plans_commitments` by query. Core items do not consume dynamic TopK. |
| Budgets | Dynamic memory and external evidence have separate limits. Keep core facts intact; reject requests when required content exceeds the context limit. |
| Representation | SQLite owns structured facts, events and indexes. `MEMORY.md` is the editable full view; `HISTORY.md` is an export, not an editing input. |
| Extraction | Retain compression and `/new` archival triggers. Recent conversation may remain only in session history until archival. |
| Runtime | Use one structured memory service and remove the legacy/v2 mode split. |
| Backends | Share DashScope embedding/reranking implementations. Dynamic memory embedding transmits those facts to the configured service. |
| Knowledge entry | The model calls `kb_search`; code handles local retrieval, evidence assessment and bounded online supplementation. |
| Online boundary | At most one search and three distinct fetched URLs per `kb_search`. Personal-memory misses do not trigger web search. |
| Knowledge storage | Pages retain source text; parents retain context; children participate in recall, reranking and citations. |
| Separation | Personal memory and web knowledge keep separate data models and stores while sharing retrieval utilities. |
| Conversion | Convert only the known workspace formats into a new offline copy; retain original data. |

## Stage order and status

| Stage | Status | Outcome |
| --- | --- | --- |
| P1 | Complete | Unified memory service, revisioned writes and prepared context |
| P2 | Complete | Validated Markdown synchronization and conflict recovery |
| P3 | Complete | Incremental semantic memory indexes and bilingual retrieval |
| P4 | Complete | Evidence selection and structured sufficiency assessment |
| P5 | Complete | One bounded online supplementation round |
| P6 | Complete | Request budgeting and offline workspace conversion |
| P7 | Planned | Real-model evaluation and parameter calibration |

## Module ownership

- `memory_db.py`: records, snapshots, events, revisions and transactional indexes.
- `memory_pipeline.py`: structured extraction without persistence side effects.
- `memory_service.py`: synchronization, retrieval, extraction and publication.
- `memory_sync.py`: strict Markdown parsing, rendering and snapshot differences.
- `retrieval.py`: shared backend clients, tokenization and rank fusion.
- `knowledge_db.py`: pages, parents, children, indexes and model metadata.
- `knowledge.py`: webpage ingestion and ranked local candidates.
- `knowledge_retrieval.py`: evidence selection, assessment and supplementation.
- `tools/web.py`: structured search results and validated fetches.
- `tools/knowledge.py`: argument validation and tool result serialization.
- `loop.py`, `context.py`, `runner.py`: shared services, prepared messages and
  the final model-request budget boundary.

Data types stay with their owning modules unless a caller actually needs a
shared definition.

## P1: unified memory runtime

Define `MemoryRecord`, `MemorySnapshot`, `MemoryContext` and `MemoryWriteResult`.
Snapshots carry revisions; write results distinguish database commit from view
export. Queries do not rebuild indexes or advance revisions. A transaction
updates events, the canonical snapshot, FTS and the revision together.

The extraction pipeline accepts the current snapshot and messages, returning a
complete replacement snapshot and history summary. `MemoryService` owns writes
and exports. Context construction and token estimation consume the same
prepared context instead of independently reading storage.

Update CLI, gateway, HTTP and SDK construction together. Advance archival
positions and clear `/new` history only after a committed write. Remove mode
branches and adapt evaluators with their callers. CI must collect both `tests/`
and `nanobot/tests/`.

Acceptance covers complete core injection, dynamic-only TopK, no duplicate
injection, transactional failures, read-only query stability and entry wiring.

## P2: Markdown synchronization

Parse the complete six-category Markdown view strictly. Preserve IDs for
unchanged facts, apply additions and deletions, and report malformed lines. A
valid empty template expresses deletion; an empty file is invalid.

Track the last published text/revision and pending publication intent. Import
valid edits before reads and writes. Detect changes made while extraction is
in flight, discard stale results and surface conflicts. Replace each exported
file atomically and recover interrupted publication without silently overwriting
new user edits. Keep `HISTORY.md` as an export.

Acceptance covers edits, deletion, invalid input, concurrent changes, missing
views and interrupted publication through service and entry-point tests.

## P3: semantic memory retrieval

Share backend clients, English token normalization, Chinese lexical tokens and
rank fusion with knowledge retrieval. Store model/dimension/tokenizer metadata
and reject incompatible indexes.

Index dynamic facts only. Embed changed or added text before committing the
corresponding snapshot; remove vectors for deleted facts. Combine lexical and
vector candidates, apply relevance thresholds and enforce the dynamic token
budget. Core facts remain intact and do not consume dynamic TopK. Avoid query
embedding when the dynamic store or query is empty.

Acceptance uses fixed embeddings and real sqlite-vec where available, including
Chinese queries, stale metadata, edit synchronization and incremental writes.

## P4: evidence selection and assessment

Separate ranked local candidates from evidence accepted for the answer. Keep
actual similarity and reranker signals. Apply relevance and document, child and
token limits to accepted evidence while retaining parent/child/source links.

Ask the configured main model for a structured coverage assessment using the
selected evidence. Distinguish insufficient evidence from retrieval or assessment
errors. Rank-fusion position alone is not proof of sufficiency. BEIR evaluation
uses the local ranking path without calling the assessment model.

Acceptance covers score filtering, evidence budgets, structured tool responses,
provider tool-choice formatting and error classification.

## P5: bounded online supplementation

Expose typed search hits while preserving the existing text tool interface.
When local evidence is insufficient, search once with the original query.
Normalize, deduplicate and validate results in order, fetch at most three valid
URLs, await ingestion, then repeat local retrieval and assessment once.

Preserve useful evidence when individual URLs fail. Validation failures do not
consume a fetch slot. Local sufficiency or service errors do not trigger an
online search. Return attempted, fetched and ingested URLs with per-URL errors.

Acceptance uses mocked search, HTTP, embeddings, reranking and assessment with
real local database writes; it does not establish real search quality.

## P6: request budgeting and offline conversion

Count prepared messages, tool definitions and reserved output immediately before
each model call, after hooks and after new tool results. Remove whole dynamic
records and whole evidence children with unreferenced parents. Invalidate an
earlier sufficiency judgment if its evidence is trimmed. Keep core memory intact;
return a context-limit error when required content cannot fit.

Provide read-only conversion preflight and explicit source selection for
ambiguous DB/Markdown state. Support memory schemas 5–8, knowledge schemas 3–4
and complete six-category Markdown workspaces. Write a new offline copy, retain
originals and rebuild FTS. Mark missing vectors as requiring rebuild; only an
explicit embedding option may send data to the configured API.

Acceptance covers ordinary/system messages, streaming, SDK/HTTP errors,
publication recovery, unchanged source bytes, preserved facts and relationships,
failed conversion and real sqlite-vec with fixed vectors.

## P7: evaluation and calibration

Use explicitly supplied datasets and separately authorized real-service runs to
measure extraction, preservation, retrieval, evidence sufficiency, online
supplementation, latency and costs. Compare declared inputs and configurations;
keep synthetic contract tests distinct from model-quality claims.

Datasets and generated reports remain outside Git history. Do not infer that
P7 has been completed from unit-test results or from implementation milestones.
