# P4 knowledge evidence contract

P4 separates local ranking from evidence sufficiency.

`WebKnowledgeService.search_local()` performs FTS/vector recall, RRF fusion, mandatory child
reranking and parent aggregation. It returns the full configured rerank candidate pool as typed
parent and child evidence, ordered by parent rank and then child rerank rank. Every child carries its parent/page IDs, source URL, partial-page flag, recall
sources and ranks, cosine similarity when available, and the raw rerank score. Scores are ranking
signals, not probabilities.

`KnowledgeRetriever.retrieve()` then:

1. removes child evidence below `knowledge.rerankRelevanceThreshold`;
2. selects parent-child evidence that fits `knowledge.evidenceTokenBudget`, applying `docLimit`,
   `evidenceLimit`, and `maxChildrenPerParent` only to accepted evidence; rejected candidates consume
   no slots, so later usable candidates remain eligible;
3. asks the configured main chat model whether the available evidence covers every part of the
   question;
4. validates the structured assessment and returns one of `sufficient`, `insufficient`,
   `retrieval_error`, or `assessment_error`.

Empty and all-low-score results are insufficient without a model call. A single relevant child can
be sufficient when it fully answers the question. Assessment timeout, provider failure, malformed
structured output, and local retrieval failure remain explicit errors. Webpage text is labelled as
untrusted data in the assessment prompt.

The Codex provider converts the shared Chat Completions tool-choice format into Responses format
for both ordinary and streaming calls. This lets the configured Codex main model provide the same
structured assessment contract.

The SQLite vector index retains its existing L2 search ordering. Cosine similarity is computed
separately from the stored vector, so the evidence field has the same meaning with sqlite-vec and
the array test backend; no index rebuild is required for this change.

BEIR evaluates the top-ranked candidate parents and uses the same `select_evidence()` function as
`retrieve()` for evidence diagnostics. Candidate ranking and the evidence actually returned are
separate collections; BEIR does not call the assessment model.

P4 does not call web search or fetch. Automated online supplementation begins in P5.

## Validation

Run the knowledge tests and `tests/providers/test_openai_codex_provider.py`.
Tests use temporary SQLite storage, fixed vectors and mocked model backends.
They cover real sqlite-vec and array similarity, candidate selection, sufficient
and insufficient assessments, and explicit failure states. Codex requests are
validated against the Responses SDK schema with mocked credentials and transport.
Live model quality and threshold calibration remain evaluation work in P7.
