# P3 personal-memory semantic retrieval

P3 replaces dynamic-memory FTS-only lookup with one production path: shared bilingual FTS and
vector recall, followed by reciprocal-rank fusion. `personal_profile`, `preferences`, and
`constraints` remain complete, unconditional core context. Only `projects`, `daily_life`, and
`plans_commitments` consume retrieval slots and the dynamic token budget.

`prepare_context()` uses `memory.dynamicTopK` when no retrieval limit is supplied (or it is
`None`). An explicit limit can tighten this cap, but cannot exceed the configured TopK.

The budget counts the actual rendered dynamic block with `cl100k_base`, including headings,
row formatting, and separators. When core memory is absent it also includes the `# Memory`
heading. Core text is excluded from this budget and is always preserved. All Unicode text and
literal tokenizer special-token strings are counted as ordinary text; no character-count estimate
is used. Candidates are considered in retrieval order; oversized entries are skipped whole.

Multi-character Chinese queries use meaningful bigrams, without independently matching single
characters. A memory candidate must pass either a lexical gate (at least 50% coverage of distinct
meaningful query terms) or the configured vector similarity gate. Single-character queries remain
supported. These are initial relevance rules; quality calibration remains part of P7.

`memory-v2-query-eval` inherits recall limits, relevance threshold, dynamic TopK, and token budget
from `config.memory`. `--top-k` explicitly overrides the evaluation TopK. JSON and Markdown reports
record the effective retrieval settings separately from the support-matching threshold. The
evaluation embedding model remains selected by `--embedding-model` and is recorded separately.

## Configuration and dependencies

Configure `memory.embedding`, `memory.ftsRecallLimit`, `memory.vectorRecallLimit`,
`memory.dynamicTopK`, `memory.dynamicTokenBudget`, and `memory.vectorSimilarityThreshold` in
`config.json`. The embedding API key is `providers.dashscope.apiKey`; semantic memory does not
depend on `knowledge.enabled`.

Install the production vector extension with:

```bash
uv sync --extra web_knowledge
```

Missing credentials or `sqlite-vec` fail explicitly. Dynamic memory text is sent to DashScope when
it is added or changed, and each non-empty dynamic query is sent for query embedding. Core memory
is never query-embedded. Empty queries and empty dynamic stores avoid embedding calls.

## Index consistency and rebuilding

`memory.db` records the embedding provider, model, dimension, vector backend, and FTS tokenizer
version. A mismatch stops startup with a rebuild instruction. Snapshot commits prepare embeddings
before opening the write transaction, recheck the P2 revision/file state, then update canonical
records, bilingual FTS rows, vector rows, events, and revision in one transaction. Unchanged records
retain their vector rows and are not embedded again.

Use [offline conversion and rebuilding](MEMORY_KNOWLEDGE_UPGRADE.md) for existing
workspaces. Preserve the original database when resolving an index mismatch.

## Focused validation

```bash
uv run --extra dev --extra web_knowledge pytest -q nanobot/tests/p3
```

The suite separates fixed-vector orchestration checks from a real `sqlite-vec` integration test.
It also covers Chinese and mixed-language tokenization in both memory and knowledge FTS indexes,
vector-only recall, threshold filtering, deletion/category movement, token budget behavior, and
metadata mismatch errors.
