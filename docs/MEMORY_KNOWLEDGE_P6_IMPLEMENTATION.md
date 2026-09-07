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
