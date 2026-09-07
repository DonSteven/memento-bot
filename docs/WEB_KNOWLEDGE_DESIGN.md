# Web Knowledge Design

## Purpose and scope

Keep the text of fetched pages in a local SQLite knowledge store and return
source-linked evidence through `kb_search`. Personal memory and external
knowledge have separate databases and responsibilities: personal memory records
facts about the user, while the knowledge store contains untrusted source text.

This is an edited English version of the external-knowledge research plan,
updated through local evidence assessment and bounded online supplementation. The implementation
uses the existing agent Hook and Tool interfaces and remains disabled by
default through `knowledge.enabled`.

## Integration boundaries

| Module | Responsibility |
| --- | --- |
| `nanobot/agent/knowledge_db.py` | SQLite schema, metadata, FTS/vector indexes and page replacement |
| `nanobot/agent/knowledge.py` | Fetch-result parsing, text preparation, embedding and retrieval |
| `nanobot/agent/knowledge_retrieval.py` | Select evidence, assess coverage and orchestrate one online supplement |
| `nanobot/agent/tools/knowledge.py` | Validate tool arguments and expose `kb_search` |
| `nanobot/agent/loop.py` | Construct the service, register its tool and schedule ingestion hooks |
| `nanobot/config/schema.py` | Define knowledge settings |

CLI and SDK construction paths pass the same knowledge configuration to the
agent loop. Built-in ingestion hooks run alongside application hooks. Background
ingestion reuses the loop's task scheduling rather than introducing a second
agent runtime.

## Storage and ingestion

The store keeps original page text, parent blocks and child chunks. Original
text is retained so evidence can be checked and derived indexes can be rebuilt.
Parent blocks preserve headings, FAQs, lists and tables. Smaller overlapping
child chunks provide focused retrieval units linked to their parent blocks.

For each successful `web_fetch` result:

1. Parse the structured result and extract URL, title, body and fetch metadata.
2. Reject invalid or empty input. Remove the transport's untrusted-content
   banner from the stored body while preserving the fact that the page is
   untrusted external material.
3. Normalize the URL and text for consistent storage and duplicate detection.
4. Compare the page's content hash with existing content. Identical content
   avoids repeated segmentation and embedding; changed content replaces the
   page's derived records.
5. Split the page into structural parent blocks and smaller child chunks, then
   embed the child chunks. Parent text remains available for grouping and FTS;
   the production vector path stores child embeddings only.
6. Persist the page, derived records and corresponding indexes as a consistent
   page snapshot.

Fetch metadata includes whether the content was truncated. A partial page is
not silently represented as a complete source. Database schema, embedding
dimensions and index contents must agree with the configured retrieval path.

## Local retrieval

The retrieval path searches child chunks with FTS and vector similarity, then
combines the rankings with reciprocal rank fusion. The configured DashScope reranker scores a bounded child pool. Parent scores use their best child plus a small coverage
bonus. Evidence selection limits both the total number of chunks and the number
from any one parent, while retaining the links back to source pages.

Each evidence item carries source identifiers, text, URL and title. Retrieval
scores describe ranking signals, not calibrated probabilities. The retriever asks the main model to assess coverage. Insufficient local
evidence triggers one search with the original query and at most three fetches.
Successful ingestion completes before local retrieval and assessment repeat.
See [online supplementation](KNOWLEDGE_P5_IMPLEMENTATION.md).

Core prompt guidance asks the agent to consult local knowledge first for
non-current factual questions. Requests for current information still require
fresh verification. Stored page contents remain evidence and must not be
treated as instructions to execute tools or override system messages.

## Backend and configuration boundaries

The knowledge backend uses asynchronous DashScope embedding and reranking APIs.
Configure `providers.dashscope.apiKey`, `knowledge.embedding` and
`knowledge.rerank`. The `web_knowledge` extra supplies sqlite-vec for local
storage. Document and query embeddings use their respective input types.
Embedding batches contain at most ten texts; response indexes, counts and
vector dimensions are validated before normalized vectors are stored.
Reranker results are aligned by their returned document indexes.

Tests inject fixed embeddings, fake rerankers and mocked HTTP responses. No
real service calls or model downloads are required for these regression tests.

SQLite extension loading belongs to the trusted backend loader. SQL values
must be parameterized, and extension paths must not originate in fetched text.
Writes update one page snapshot at a time. Expensive text processing should not
hold a database write transaction open.

## Capacity and maintenance

The research plan used roughly ten thousand documents as a planning scale, not
as a measured capacity result. Storage depends on document length, overlap and
embedding dimensions. Measure page, chunk and vector counts alongside database
size before changing index layout or adding another storage service.

Track ingestion failures, duplicate skips, candidate counts and retrieval
latency. Page content is the basis for rebuilding derived indexes after a
supported configuration change. Backups must account for SQLite's active write
state; copying only a live database file is not a complete backup procedure.

## Verification and evaluation

Use fictional pages and temporary databases to test:

- Configuration propagation and disabled-by-default behavior.
- Invalid fetch results, partial pages and empty text.
- Duplicate detection and replacement of changed pages.
- FTS/vector result fusion and bounded evidence selection.
- Index rebuilds and incompatible backend metadata.
- Tool registration, background scheduling and hook ordering.

Offline retrieval evaluation can measure Recall@k, MRR and nDCG against supplied
relevance judgments. It should use the production retrieval path and distinguish
document rankings from individual evidence chunks. Synthetic tests verify the
metric implementation; they do not establish benchmark quality.

Evaluation corpora, relevance judgments and generated reports are local inputs
and outputs, not repository content. Store them outside tracked source files.

## Evidence selection and assessment

Local search returns typed parent and child candidates. A shared selector applies
relevance, count and token limits before the configured main model assesses
coverage. Retrieval and assessment failures are explicit states. See
[the P4 contract](KNOWLEDGE_P4_IMPLEMENTATION.md).
