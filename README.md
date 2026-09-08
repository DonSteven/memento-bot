# Memento Bot

Memento Bot is an independently maintained extension of [HKUDS/nanobot](https://github.com/HKUDS/nanobot), focused on persistent memory, evidence-aware knowledge retrieval, and context-efficient agent workflows. It gives a personal AI assistant structured recall across conversations and reusable web evidence with explicit coverage and failure states.

**Upstream:** HKUDS/nanobot · **Baseline:** [`2dac322b2e04e5791a62da01d216cb9224ee8996`](https://github.com/DonSteven/memento-bot/commit/2dac322b2e04e5791a62da01d216cb9224ee8996) (`upstream-base`)

## What I Changed

Development after the baseline concentrates on memory, retrieval and runtime integration:

| Problem | Implemented / redesigned in Memento Bot |
| --- | --- |
| Long-lived facts need consistent updates and manual editing. | Replaced Markdown-only memory persistence with transactional SQLite snapshots, six fact categories, revision checks and recoverable Markdown synchronization. |
| Relevant facts may use different wording or Chinese text. | Added shared English/Chinese FTS5 and vector retrieval with reciprocal-rank fusion; preserve core facts while selecting dynamic facts within a token budget. |
| Similar passages alone do not establish answer coverage. | Built parent–child knowledge retrieval, reranking, source-linked evidence selection and structured model assessment of missing information. |
| Local evidence can be incomplete. | Added one bounded search-and-ingestion round, followed by retrieval and reassessment against the same persistent knowledge store. |
| Growing memory and tool results can exceed model limits. | Added model-call boundary checks, whole-record/evidence reduction, explicit context-limit results and offline workspace conversion. |

**Inherited from nanobot:** the agent loop and runner foundation, provider integrations, tools and MCP, chat channels, sessions, scheduling, CLI, Python SDK and HTTP API. Memento Bot extends these entry points; it does not claim their original implementation.

[Quick Start](#quick-start) · [Architecture](#architecture) · [Evaluation](#evaluation--reliability) · [Configuration](docs/CONFIGURATION.md)

## Why Memento Bot

A lightweight assistant needs to retain durable preferences without loading every past detail, and distinguish useful retrieved text from evidence that addresses a question. This project separates personal facts from external knowledge, makes persistence and retrieval explicit, and controls what reaches the answer model.

## Architecture

The inherited agent shell remains in use. The original diagram shows that foundation; the added memory and evidence paths are described below.

<img src="nanobot_arch.png" alt="Inherited nanobot architecture: chat apps, agent loop, tools, memory and skills" width="720">

1. **Prepare personal context:** CLI, gateway, SDK and HTTP requests reach `AgentLoop`. `MemoryService` synchronizes manual edits and returns complete core facts plus relevant dynamic facts from SQLite.
2. **Run the agent:** the runner combines prepared memory, instructions, history and tools. Request budgeting reserves output tokens and checks the request before each answer-model call, including after hooks and tool results.
3. **Retrieve knowledge when requested:** with knowledge enabled, the model can call `kb_search`. Local FTS/vector recall → reranking → evidence selection → coverage assessment. Only insufficient evidence triggers one online search, up to three validated URL fetches, ingestion, and another local retrieval/assessment.
4. **Return and retain:** source-linked evidence becomes a tool result for answer generation. Memory consolidation extracts structured snapshots; successful web fetches can populate the separate knowledge store through a hook.

Knowledge search is a tool-driven branch, not a mandatory stage of every request. Personal-memory misses do not trigger online supplementation.

## Key Extensions

### Structured and Semantic Memory

[`MemoryService`](nanobot/agent/memory_service.py) coordinates extraction, persistence, synchronization and recall. Core categories (`personal_profile`, `preferences`, `constraints`) remain complete; dynamic categories (`projects`, `daily_life`, `plans_commitments`) use lexical/vector recall, relevance gates and reciprocal-rank fusion. Selection counts the rendered dynamic block and skips oversized facts whole.

SQLite stores canonical facts and history events. Revision checks reject stale extraction results; pending publication state allows interrupted Markdown exports to recover. See [synchronization](spec/MEMORY_P2_IMPLEMENTATION.md) and [semantic retrieval](spec/MEMORY_P3_IMPLEMENTATION.md).

### Knowledge Retrieval and Evidence Assessment

[`WebKnowledgeService`](nanobot/agent/knowledge.py) splits page text into parent context blocks and child passages, indexes child vectors, and combines lexical/vector recall with reranking. [`KnowledgeRetriever`](nanobot/agent/knowledge_retrieval.py) selects evidence within relevance, count and token limits, then asks the configured main model to report sufficiency, missing points and a reason.

Results preserve URLs, parent–child relationships, scores and partial-page flags. Retrieval and assessment failures are explicit states. Rerank scores are not confidence probabilities; model-assessed sufficiency is not a factual-correctness guarantee. See the [evidence contract](spec/KNOWLEDGE_P4_IMPLEMENTATION.md).

### Bounded Online Supplementation

Insufficient local evidence permits one search-interface call using the original query and configured provider. The retriever validates and deduplicates URLs, fetches at most `onlineMaxUrls` (1–3), awaits ingestion, then repeats local retrieval and assessment. Search snippets select URLs; evidence comes from stored page bodies. Partial failures retain usable evidence and report individual errors. See [online behavior](spec/KNOWLEDGE_P5_IMPLEMENTATION.md).

### Context Budgeting and Offline Workspace Conversion

[`context_budget.py`](nanobot/agent/context_budget.py) counts prepared messages and tools with an output reserve, removes whole dynamic facts before knowledge evidence, and preserves core memory. Reducing evidence invalidates its prior coverage conclusion. Requests that still exceed the configured window stop with `context_limit`, exposed through the SDK and HTTP 400. Provider counters or local estimates are used; exact agreement with every provider's accounting is not guaranteed.

The [conversion utility](scripts/upgrade_memory_knowledge.py) defaults to read-only preflight and converts supported memory/knowledge formats into a fresh copy. It preserves originals and requires explicit source selection for conflicts. A copy without rebuilt vectors remains `REBUILD_REQUIRED`. See [conversion and switching](spec/MEMORY_KNOWLEDGE_UPGRADE.md).

Storage and conversion can be local/offline; runtime semantic retrieval uses DashScope APIs, and knowledge search also uses reranking and model assessment.

## Evaluation / Reliability

The repository includes executable validation and evaluation tools:

- **Memory contracts:** [service and persistence](nanobot/tests/p1), [manual-edit conflicts and recovery](nanobot/tests/p2), and [semantic recall and budgets](nanobot/tests/p3).
- **Knowledge contracts:** [retrieval, evidence selection, online ingestion and failures](nanobot/tests/knowledge), using temporary databases and mocked services or fixed vectors.
- **Runtime integration:** [context limits, entry-point consistency and conversion](nanobot/tests/p6), including source preservation and interrupted-publication handling.
- **Evaluation tooling:** [memory extraction](nanobot/agent/memory_semantic_eval.py), [snapshot/replay queries](nanobot/agent/memory_query_eval.py), and [BEIR retrieval](nanobot/agent/knowledge_beir_eval.py). BEIR reports ranking metrics, evidence diagnostics and query-search latency; it does not evaluate the assessment model.

No published benchmark scores or live end-to-end quality/cost results are claimed here. Tests establish local behavior; retrieval thresholds and live assessment quality still need calibration. Evaluation commands and optional dependencies are in the [CLI reference](docs/CONFIGURATION.md#-cli-reference).

Run focused local contract tests after installing development dependencies:

```bash
python -m pip install -e '.[dev,web_knowledge]'
python -m pytest nanobot/tests/p1 nanobot/tests/p2 nanobot/tests/p3 nanobot/tests/knowledge nanobot/tests/p6 -q
```

## Quick Start

Requires Python 3.11+, SQLite with extension-loading support, a chat-model API key, and a DashScope API key for semantic memory. Install **this source repository**; installing `nanobot-ai` from PyPI alone does not provide Memento Bot extensions. The Python package, executable and `~/.nanobot` paths keep their existing names.

```bash
git clone https://github.com/DonSteven/memento-bot.git
cd memento-bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[web_knowledge]'
nanobot onboard
```

For a fresh workspace, merge this into `~/.nanobot/config.json`, replacing placeholders with credentials and a model available through your chat provider:

```json
{
  "providers": {
    "openrouter": {"apiKey": "YOUR_CHAT_API_KEY"},
    "dashscope": {"apiKey": "YOUR_DASHSCOPE_API_KEY"}
  },
  "agents": {
    "defaults": {
      "provider": "openrouter",
      "model": "YOUR_OPENROUTER_MODEL_ID"
    }
  },
  "knowledge": {"enabled": true},
  "tools": {"web": {"search": {"provider": "duckduckgo"}}}
}
```

Memory defaults to DashScope `text-embedding-v4`; knowledge has its own embedding settings and uses `qwen3-rerank`. Ensure your key can access the configured endpoints/models. Set `agents.defaults.contextWindowTokens` and `maxTokens` to match your chat model. See [model settings](docs/CONFIGURATION.md#semantic-memory-and-external-web-knowledge-models) for endpoints and limits.

```bash
nanobot agent -m "What can you help me with?"
nanobot agent
```

With knowledge enabled, ask the agent to use `kb_search` for a question requiring external evidence. Chat, embeddings, reranking and assessment can incur API charges; online supplementation sends queries to the search provider and fetches external pages. Set `knowledge.enabled` to `false` to omit the knowledge tool; semantic memory still requires its vector dependency and embedding configuration.

For an existing workspace, follow [conversion and switching](spec/MEMORY_KNOWLEDGE_UPGRADE.md) before starting the runtime. For channels, MCP, Docker, services and other providers, use the [configuration and operations reference](docs/CONFIGURATION.md). The [SDK guide](docs/PYTHON_SDK.md) covers programmatic use.

## Project Structure

| Path | Responsibility |
| --- | --- |
| `nanobot/agent/memory*.py` | Structured memory, synchronization, retrieval and evaluation |
| `nanobot/agent/knowledge*.py` | Page storage, parent–child retrieval, evidence assessment and BEIR evaluation |
| `nanobot/agent/retrieval.py` | Shared lexical/vector primitives and DashScope backends |
| `nanobot/agent/context_budget.py` | Request sizing and evidence/memory reduction |
| `nanobot/agent/loop.py`, `nanobot/agent/runner.py` | Agent orchestration and model-call boundary |
| `nanobot/cli/`, `nanobot/api/`, `nanobot/nanobot.py` | CLI, HTTP and SDK entry points |
| `nanobot/channels/`, `nanobot/providers/`, `nanobot/agent/tools/` | Inherited integration foundations, with targeted extensions |
| `nanobot/tests/`, `tests/` | Extension contracts and inherited/updated runtime tests |
| `scripts/`, `spec/`, `docs/` | Conversion utility, design details and operating reference |

## Upstream & Attribution

Memento Bot is an independently maintained derivative of [HKUDS/nanobot](https://github.com/HKUDS/nanobot), with substantial modifications and extensions. It is not affiliated with HKUDS.

Development in this repository diverged from upstream at `2dac322b2e04e5791a62da01d216cb9224ee8996`. The annotated Git tag `upstream-base` identifies that baseline. The original nanobot code is distributed under the MIT License; original copyright and permission notices are preserved. Subsequent Memento Bot extensions are maintained in this repository.

[Contribution guide](CONTRIBUTING.md) · [Security policy](SECURITY.md)

## License

[MIT](LICENSE). Original notice: `Copyright (c) 2025 nanobot contributors`.
