# P6 Entry Integration and Context Limits

CLI agent, gateway and serve, the Python SDK and HTTP requests use the same
configured `AgentLoop`. Ordinary and system messages prepare memory through the
same service and pass that `MemoryContext` into `AgentRunSpec`. Core and selected
dynamic facts each appear once in the final messages.

## Request budgeting

`agent/context_budget.py` counts prepared messages, tool definitions and reserved
output tokens against the configured context window. It first removes whole
dynamic-memory records from the end. Oversized tool evidence is reduced by whole
children and then unreferenced parents, preserving source relationships. Reduced
evidence is marked `budget_limited`; its previous sufficiency conclusion is
invalidated. Core memory is never truncated, summarized or removed.

Mandatory content is checked before history compression. If it already exceeds
the budget, neither extraction nor answer generation runs. Existing history
compression remains in use; budgeting introduces no additional retrieval or
compression-model call. Counting uses the prepared context without reading
memory again or embedding the query again.

`AgentRunner` checks after hooks and immediately before each streaming or ordinary
model call, including after tool results are appended. An oversized request ends
with `stop_reason=context_limit`. Streaming sends the error text and closes with
`resuming=False`; the iteration-end hook receives the limit state. Ordinary and
system messages return a clear explanation. SDK `RunResult.stop_reason` exposes
the state, and HTTP returns status 400 with `error.type=context_limit`.

The local token counter treats literal tokenizer special-token strings as text
and includes non-text multimodal payloads instead of silently omitting them.
It prefers the provider counter and otherwise uses a local tokenizer estimate.
This check can stop known oversized requests; it does not promise exact agreement
with every provider's server-side or multimodal token accounting.

## Validation

`nanobot/tests/p6/test_context_budget.py` and `test_entry_integration.py` use
prepared memory, fixed vectors and mocked external services in temporary
workspaces. They cover entry consistency, SDK/HTTP limits, streaming failure
propagation, one query embedding, archive and manual-edit flows, online knowledge
ingestion followed by local reuse, and recovery after interrupted publication.
Related memory, knowledge, runner and configuration tests exercise the callers.
These tests validate local contracts, not live retrieval quality or API cost.

## Offline workspace conversion

`scripts/upgrade_memory_knowledge.py` defaults to read-only preflight. It supports
memory schemas v5–v8, knowledge schemas v3–v4 and complete six-category Markdown
workspaces. Conversion writes a new offline copy, preserves originals and
rebuilds derived FTS indexes. Ambiguous facts require an explicit source choice;
history cannot resurrect deleted memories. Invalid formats and unsafe output
symlinks are rejected before publication.

Without embeddings, required vector indexes remain `REBUILD_REQUIRED` and the
copy is not ready for runtime use. An explicit rebuild option can use the target
embedding backend. See [conversion and switching](MEMORY_KNOWLEDGE_UPGRADE.md)
for source selection, configuration, stopped writers, data transfer and switching.

`test_upgrade.py` uses temporary copies and mocked embeddings with real sqlite-vec.
It compares source bytes, canonical fields, event fields, page provenance and
parent-child links; checks Chinese FTS and fixed-vector recall; and verifies that
conflicts or failures do not publish a partial destination or change source data.
Production conversion, paid embeddings and service switching are separate actions.
