# P5 Online Knowledge Supplementation

`KnowledgeRetriever` starts with local retrieval, evidence selection and structured
coverage assessment. Sufficient evidence and local retrieval or assessment errors
return immediately. An empty query is rejected. Only an insufficient result starts
one round of online supplementation using the original query.

## Search, fetch and ingestion

`WebSearchTool.search()` returns `WebSearchHit(title, url, snippet)` records from
all five supported search providers. `execute()` retains the human-readable tool
output and error presentation. Search failures propagate to the orchestrator.

Search results retain provider order. URLs are normalized, fragments removed,
duplicates discarded and existing URL safety checks applied. Invalid URLs and
validation exceptions are recorded individually; later valid URLs remain eligible.
`knowledge.onlineMaxUrls` defaults to 3 and accepts values from 1 through 3. Invalid
candidates do not consume a fetch slot. There is only one search round.

The retriever uses the configured `WebFetchTool` instance. Ingestion rejects error
results, failed HTTP responses, empty or non-string bodies, images and other binary
content. MIME matching ignores case and parameters. XHTML uses body extraction;
XML and Atom content can remain text. Partial-page metadata is preserved.

The internal flow awaits `ingest_page()`, including chunking, embeddings and the
SQLite transaction. It shares this entry point with `WebKnowledgeHook`. Internal
fetches call the tool instance directly, so they do not create a model tool-call
record or trigger duplicate background ingestion. Already stored identical pages
can be reused without embedding them again.

After successful ingestion, the retriever repeats local retrieval and assessment
with the same query and limits. New and existing evidence use the same index,
reranker and selection rules. Search snippets select URLs only; returned evidence
comes from stored page bodies and retains parent-child links, URLs, scores and
partial-page flags.

## Results and errors

- `online_attempted` is false when local evidence is sufficient or local retrieval
  or assessment fails. It becomes true when the one search call is attempted.
- `fetched_urls` lists actual fetch attempts. `ingested_urls` lists successful
  ingestion returns, including unchanged pages already present in storage.
- No search results leave the result `insufficient`. A result still lacking
  coverage after supplementation retains `missing_points` without another search.
- Failed online execution with no successful ingestion returns `online_error`
  and preserves the original evidence and insufficiency assessment.
- Partial success retains the final retrieval/assessment status and reports
  individual failures in `online_errors`.
- Retrieval and assessment failures use `sufficient=null`. If the second local
  retrieval fails, the first selected evidence remains available.

`AgentLoop` registers and injects the same configured search and fetch instances,
including provider and proxy settings. `kb_search` invokes the complete retrieval
flow. Personal-memory retrieval has no added online search path.

The query is sent to the configured search service. Fetched text can be sent to
the embedding, reranking and main-model assessment services, incurring their API
charges. One round means one search-interface call; existing Jina/readability
extraction behavior remains part of the fetcher.

## Validation

Run `nanobot/tests/knowledge`, `tests/tools/test_web_search_tool.py` and
`tests/tools/test_web_fetch_security.py`, together with the affected memory,
context and provider tests. These tests use temporary SQLite storage, fixed
vectors and mocked HTTP, DNS, search, embedding, reranking and assessment backends.
They cover invalid URLs before and after successful fetches, partial failures,
cold-store persistence, repeat retrieval and XML-family text extraction.

These contract tests do not measure live model quality, threshold calibration,
search quality or cost. Those remain evaluation work in P7. Workspace conversion
and end-to-end input budgeting are addressed in P6.
