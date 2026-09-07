# Offline Memory and Knowledge Workspace Conversion

The converter supports memory SQLite schemas v5–v8 with canonical facts and raw
events, knowledge schemas v3–v4 with page/parent/child records, and complete
six-category `MEMORY.md` workspaces without a database. Other schemas, arbitrary
Markdown and invalid entries require manual preparation. The tool never asks a
model to infer missing facts.

## Read-only preflight

From the repository root, run:

```bash
.venv/bin/python scripts/upgrade_memory_knowledge.py /path/to/source-workspace
```

The default command reports schema versions, record counts, differences between
database and Markdown facts, interrupted publication, required index rebuilds and
target configuration changes. It does not initialize databases, construct runtime
services or call an embedding backend.

Without a publication baseline, differing database and Markdown facts require an
explicit `--memory-source database` or `--memory-source markdown` choice. Deleted
facts cannot be restored automatically from history. A valid empty six-category
template clears memory; a zero-byte or malformed file is rejected with its line
number where applicable.

With a baseline, unchanged Markdown selects the database. File-only changes
select Markdown; changes on both sides require a source choice. During interrupted
publication, the expected old text or committed target text selects the committed
database. An additional human edit produces a conflict.

## Target configuration

Use a separate target JSON configuration and remove `memory.mode`. Set provider,
model and dimensions under `memory.embedding` and `knowledge.embedding`, and valid
`knowledge.rerank` settings when knowledge is enabled. Runtime services use
DashScope credentials; `uv sync --extra web_knowledge` installs sqlite-vec.
The current `Config` validates target settings strictly without silently
converting obsolete fields. Keep credentials out of public reports.

`agents.defaults.contextWindowTokens` is the model's combined input/output limit;
`maxTokens` reserves output space. Dynamic-memory TopK/token limits and knowledge
evidence budgets remain separately configurable.

## Convert a copy without API calls

Stop every writer to the source workspace, including CLI, gateway, serve and SDK
processes, before running:

```bash
.venv/bin/python scripts/upgrade_memory_knowledge.py /path/to/source-workspace \
  --destination /path/to/offline-copy --source-stopped \
  --config /path/to/target-config.json
```

Add the chosen `--memory-source` value if preflight requires it. The destination
must not exist and must lie outside the source. Conversion builds a temporary
copy and publishes it only on success. Failure leaves the source unchanged and
publishes no partial destination. Storage-directory symlinks and symlinks at
`upgrade-originals` or `upgrade-report.json`, including dangling links, are
rejected before writing. SQLite backups include committed WAL contents.

The copy preserves memory IDs, all raw-event fields, page bodies, URLs, partial
flags, timestamps and parent-child relationships. `upgrade-originals/source-*`
retains original SQLite backups and Markdown views; other workspace files are
also copied. Database events remain authoritative history. `HISTORY.md` cannot
recreate deleted facts. In a Markdown-only workspace, the original history is
preserved as one imported event.

FTS indexes are rebuilt using the current tokenizer. Without embeddings, stores
containing dynamic facts or knowledge children are marked `REBUILD_REQUIRED`.
The report has `ready=false` and lists `pending_embeddings`. Such a copy is not
ready to start as a runtime workspace.

## Explicit embedding rebuild

Rebuilding sends all selected dynamic-memory text and all knowledge-child text
to the configured DashScope embedding API and may incur charges. It does not
embed core memory or run online web search. Review the preflight counts and
obtain authorization for that data transfer and cost before running:

```bash
.venv/bin/python scripts/upgrade_memory_knowledge.py /path/to/offline-copy \
  --destination /path/to/ready-copy --source-stopped \
  --config /path/to/target-config.json --rebuild-embeddings
```

This creates another copy and retains the earlier original backups. Conversion
and rebuilding can also start directly from the stopped original workspace.
The tool does not change source configuration or redirect running services.
Require `ready=true` and `pending_embeddings=[]` before switching.

## Switch the runtime separately

1. Keep the source stopped after the conversion snapshot. If writing resumed,
   recreate the copy from the latest source instead of selecting stale data.
2. Check counts, source choice, core/dynamic facts, events, page provenance and
   recovery of any pending publication.
3. Verify ready indexes and matching target model/dimensions. Preserve the
   original data and configuration as recoverable copies.
4. Point CLI, gateway, serve and SDK at the ready workspace and target config.
   Start one entry point and verify core injection, dynamic retrieval and local
   `kb_search`, then resume other entry points.
5. Before rollback, stop new writes and preserve newly generated data for manual
   reconciliation; do not overwrite either side.

Tests use temporary copies, mocked embeddings and real sqlite-vec with fixed
vectors. They do not perform a production workspace switch or paid rebuild.
