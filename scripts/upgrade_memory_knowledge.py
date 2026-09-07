"""One-time conversion of known memory v5–v8 / knowledge v3–v4 workspaces.

Default is read-only preflight. Conversion requires a fresh destination and a stopped source.
Embedding is opt-in; without it the copy is explicitly marked as requiring index rebuild.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

# Support invocation as a script from the repository checkout.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobot.agent.knowledge_db import WebKnowledgeDatabase
from nanobot.agent.memory_db import (
    DYNAMIC_MEMORY_CLASSES,
    MemoryDatabase,
    MemoryRecord,
    validate_memory_fields,
)
from nanobot.agent.memory_sync import (
    parse_memory_markdown,
    render_memory_markdown,
)
from nanobot.agent.retrieval import DashScopeEmbeddingBackend, fts_index_text, vector_json
from nanobot.config.schema import Config

MEMORY_PATH = Path("memory/memory.db")
KNOWLEDGE_PATH = Path("knowledge/web_knowledge.db")


def read_database(path: Path, tables: tuple[str, ...]) -> dict:
    if not path.exists():
        return {}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        names = {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
        result = {table: [dict(row) for row in conn.execute(f'SELECT * FROM "{table}"')]
                  for table in tables if table in names}
        result["metadata"] = dict(conn.execute("select key, value from metadata"))
        return result


def inspect_workspace(source: Path) -> dict:
    """Read authoritative rows without instantiating runtime DBs or modifying files."""
    if not source.is_dir():
        raise ValueError("Source workspace does not exist")
    memory = read_database(source / MEMORY_PATH, ("canonical_memories", "raw_events", "memory_sync_state"))
    knowledge = read_database(source / KNOWLEDGE_PATH, ("web_pages", "page_parents", "parent_children"))
    problems: list[str] = []
    for name, data, versions in (("memory", memory, {5, 6, 7, 8}), ("knowledge", knowledge, {3, 4})):
        if data and int(data["metadata"].get("schema_version", -1)) not in versions:
            problems.append(f"Unsupported {name} schema: {data['metadata'].get('schema_version')}")
    records = []
    for row in memory.get("canonical_memories", []):
        try:
            validate_memory_fields(row["main_class"], row["sub_class"], row["text"])
            records.append(MemoryRecord(**row))
        except (ValueError, TypeError, KeyError) as exc:
            problems.append(f"Invalid database memory: {exc}")
    markdown_path = source / "memory/MEMORY.md"
    text = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else None
    markdown_records = None
    if text is not None:
        try:
            markdown_records = parse_memory_markdown(text)
        except ValueError as exc:
            problems.append(str(exc))
    state = next(iter(memory.get("memory_sync_state", [])), {})
    def keys(items):
        return {(r.main_class, r.sub_class, r.text) for r in items}
    differs = markdown_records is not None and keys(records) != keys(markdown_records)
    selected = "database" if memory else "markdown"
    conflict = False
    if state.get("pending_revision") is not None:
        conflict = text not in (state.get("pending_expected_text"), state.get("pending_target_text"))
        selected = "database"
    elif differs and memory:
        if state.get("published_revision") is None:
            conflict = True
        elif text == state.get("published_text"):
            selected = "database"
        elif int(memory["metadata"].get("revision", 0)) == state["published_revision"]:
            selected = "markdown"
        else:
            conflict = True
    if markdown_records is None and not memory and text is None:
        selected = "database"
    for table in ("web_pages", "page_parents", "parent_children"):
        if knowledge and table not in knowledge:
            problems.append(f"Missing knowledge table: {table}")
    if memory and any(table not in memory for table in ("canonical_memories", "raw_events")):
        problems.append("Missing authoritative memory tables")
    return {"memory": memory, "knowledge": knowledge, "records": tuple(records),
            "markdown_records": markdown_records, "text": text, "selected_source": selected,
            "report": {"memory_schema": memory.get("metadata", {}).get("schema_version"),
                       "knowledge_schema": knowledge.get("metadata", {}).get("schema_version"),
                       "memories": len(records), "markdown_memories": len(markdown_records or ()),
                       "events": len(memory.get("raw_events", [])),
                       "pages": len(knowledge.get("web_pages", [])),
                       "parents": len(knowledge.get("page_parents", [])),
                       "children": len(knowledge.get("parent_children", [])),
                       "db_only": sorted(keys(records) - keys(markdown_records or ())),
                       "markdown_only": sorted(keys(markdown_records or ()) - keys(records)),
                       "dynamic_memories": sum(r.main_class in DYNAMIC_MEMORY_CLASSES for r in records),
                       "pending_publish": state.get("pending_revision") is not None,
                       "db_markdown_differ": differs, "source_choice_required": conflict,
                       "selected_source": None if conflict else selected, "problems": problems,
                       "rebuild_required": ["memory FTS/vector", "knowledge FTS/vector"],
                       "configuration_changes": ["Remove memory.mode", "Configure memory.embedding",
                                                 "Enable knowledge.rerank when knowledge is enabled",
                                                 "Set contextWindowTokens to the model input+output limit"]}}


def insert_rows(conn, table, rows):
    for row in rows:
        columns = list(row)
        # Names originate only in known SQLite schemas; reject extra/unsupported columns.
        allowed = {r[1] for r in conn.execute(f'pragma table_info("{table}")')}
        if not set(columns) <= allowed:
            raise ValueError(f"Unsupported columns in {table}: {set(columns) - allowed}")
        conn.execute(f'insert into "{table}" ({",".join(columns)}) values ({",".join("?" for _ in columns)})',
                     [row[key] for key in columns])


async def convert_workspace(source: Path, destination: Path, *, config: Config,
                            memory_source: str | None = None, memory_embedder=None,
                            knowledge_embedder=None, vec_backend="sqlite-vec") -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("Destination must be new and outside the source workspace")
    data = inspect_workspace(source)
    report = data["report"]
    if report["problems"]:
        raise ValueError("; ".join(report["problems"]))
    if report["source_choice_required"] and memory_source is None:
        raise ValueError("DB/Markdown conflict: choose --memory-source database or markdown")
    choice = memory_source or data["selected_source"]
    if choice not in {"database", "markdown"}:
        raise ValueError("Unknown memory source")
    if choice == "markdown" and data["markdown_records"] is None:
        raise ValueError("Selected Markdown source is missing")
    records = data["records"] if choice == "database" else data["markdown_records"]
    # Preserve IDs for facts retained across an explicit Markdown selection.
    existing = {(r.main_class, r.sub_class, r.text): r for r in data["records"]}
    records = tuple(existing.get((r.main_class, r.sub_class, r.text), r) for r in records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nanobot-upgrade-", dir=destination.parent) as temp:
        target = Path(temp) / "workspace"
        shutil.copytree(source, target, symlinks=True)
        # Refuse symlinks in owned storage before any writes could escape the copy.
        for name in ("upgrade-originals", "upgrade-report.json"):
            if (target / name).is_symlink():
                raise ValueError("Upgrade output paths must not be symlinks")
        for directory in (target / "memory", target / "knowledge"):
            if directory.is_symlink() or (directory.exists() and any(p.is_symlink() for p in directory.rglob("*"))):
                raise ValueError("Memory/knowledge storage must not contain symlinks")
        originals = target / "upgrade-originals"
        originals.mkdir(exist_ok=True)
        originals = Path(tempfile.mkdtemp(prefix="source-", dir=originals))
        for relative in (MEMORY_PATH, KNOWLEDGE_PATH):
            copied = target / relative
            if copied.exists():
                saved = originals / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                # SQLite backup also captures committed WAL content.
                with closing(sqlite3.connect((source / relative).as_uri() + "?mode=ro", uri=True)) as src:
                    with closing(sqlite3.connect(saved)) as dst:
                        src.backup(dst)
                for suffix in ("", "-wal", "-shm"):
                    Path(str(copied) + suffix).unlink(missing_ok=True)
        for filename in ("MEMORY.md", "HISTORY.md"):
            old = target / "memory" / filename
            if old.exists():
                shutil.copy2(old, originals / filename)
        mc = config.memory.embedding
        db = MemoryDatabase(target, vec_backend=vec_backend)
        db.initialize(mc.dimensions, embedding_provider=mc.provider, embedding_model=mc.model)
        events = list(data["memory"].get("raw_events", []))
        if not data["memory"] and (originals / "HISTORY.md").exists():
            history = (originals / "HISTORY.md").read_text(encoding="utf-8")
            if history:
                events.append({"event_id": "imported-history", "ts": "", "session_key": "import",
                               "history_text": history, "plain_text": history})
        with db.connect() as conn:
            insert_rows(conn, "canonical_memories", [dict(memory_id=r.memory_id, main_class=r.main_class,
                        sub_class=r.sub_class, text=r.text) for r in records])
            insert_rows(conn, "raw_events", events)
            revision = int(data["memory"].get("metadata", {}).get("revision", 0))
            if choice == "markdown" and report["db_markdown_differ"]:
                revision += 1
            conn.execute("update metadata set value=? where key='revision'", (str(revision),))
        dynamic = [r for r in records if r.main_class in DYNAMIC_MEMORY_CLASSES]
        pending = []
        if memory_embedder is not None or not dynamic:
            vectors = await memory_embedder.embed_documents([r.text for r in dynamic]) if dynamic else []
            if len(vectors) != len(dynamic):
                raise ValueError("Memory embedding count mismatch")
            db.rebuild_indexes(dynamic_embeddings=dict(zip([r.memory_id for r in dynamic], vectors)),
                               vector_dim=mc.dimensions, embedding_provider=mc.provider, embedding_model=mc.model)
        else:
            with db.connect() as conn:
                for r in dynamic:
                    conn.execute("insert into canonical_fts(memory_id,tokens) values (?,?)",
                                 (r.memory_id, fts_index_text(f"{r.sub_class} {r.text}")))
                conn.execute("update metadata set value='REBUILD_REQUIRED' where key='embedding_model'")
            pending.append("memory")
        db.memory_file.write_text(render_memory_markdown(records), encoding="utf-8")
        db.record_memory_publish_baseline(revision=revision, text=render_memory_markdown(records))
        db.history_file.write_text(db.render_history_view(), encoding="utf-8")
        if data["knowledge"]:
            kc = config.knowledge.embedding
            kb = WebKnowledgeDatabase(target, vec_backend=vec_backend)
            kb.initialize(kc.dimensions, embedding_provider=kc.provider, embedding_model=kc.model)
            with kb.connect() as conn:
                for table in ("web_pages", "page_parents", "parent_children"):
                    insert_rows(conn, table, data["knowledge"][table])
                if conn.execute("pragma foreign_key_check").fetchone():
                    raise ValueError("Knowledge parent/child references are invalid")
            kb.rebuild_indexes()
            children = data["knowledge"]["parent_children"]
            if knowledge_embedder is not None or not children:
                vectors = await knowledge_embedder.embed_documents([c["text"] for c in children]) if children else []
                if len(vectors) != len(children) or any(len(v) != kc.dimensions for v in vectors):
                    raise ValueError("Knowledge embedding count/dimension mismatch")
                column = "embedding" if vec_backend == "sqlite-vec" else "embedding_json"
                with kb.connect() as conn:
                    for child, vector in zip(children, vectors):
                        conn.execute(f"insert into parent_child_vec(rowid,{column}) values (?,?)",
                                     (child["child_id"], vector_json(vector)))
            else:
                with kb.connect() as conn:
                    conn.execute("update metadata set value='REBUILD_REQUIRED' where key='embedding_model'")
                pending.append("knowledge")
        report = {**report, "selected_source": choice, "source_choice_required": False,
                  "source_memories": report["memories"], "memories": len(records), "events": len(events),
                  "rebuild_required": [f"{name} vector" for name in pending],
                  "pending_embeddings": pending,
                  "ready": not pending, "destination": str(destination)}
        (target / "upgrade-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        target.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--memory-source", choices=("database", "markdown"))
    parser.add_argument("--source-stopped", action="store_true", help="Confirm all writers to the source are stopped")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--rebuild-embeddings", action="store_true", help="Send dynamic memory and knowledge children to configured paid embedding APIs")
    args = parser.parse_args()
    if not args.destination:
        print(json.dumps(inspect_workspace(args.source)["report"], ensure_ascii=False, indent=2))
        return
    if not args.source_stopped:
        parser.error("Stop CLI/gateway/serve/SDK writers, then pass --source-stopped")
    if not args.config:
        parser.error("Conversion requires an explicit target --config")
    config = Config.model_validate_json(args.config.read_text(encoding="utf-8"))
    embedders = {}
    if args.rebuild_embeddings:
        key = config.providers.dashscope.api_key
        for name in ("memory", "knowledge"):
            ec = getattr(config, name).embedding
            embedders[f"{name}_embedder"] = DashScopeEmbeddingBackend(
                api_key=key, api_base=ec.api_base, model=ec.model, dimension=ec.dimensions)
    print(json.dumps(asyncio.run(convert_workspace(args.source, args.destination, config=config,
                     memory_source=args.memory_source, **embedders)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
