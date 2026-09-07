"""BEIR benchmark runner for the external web knowledge retrieval stack."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlsplit

from nanobot.agent.knowledge import WebKnowledgeService
from nanobot.agent.knowledge_db import WebKnowledgeDatabase
from nanobot.agent.knowledge_retrieval import select_evidence
from nanobot.config.schema import KnowledgeConfig

REPORT_VERSION = 2
DEFAULT_DATASET = "scifact"
DEFAULT_DOC_LIMIT = 10
DEFAULT_EVIDENCE_LIMIT = 5
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
_SUPPORTED_DATASETS = {DEFAULT_DATASET}
_DEFAULT_DATASET_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
}
_METRIC_CUTOFFS = [1, 3, 5, 10]
_MRR_CUTOFFS = [10]
_SLOWEST_QUERY_COUNT = 5


class EmbeddingBackend(Protocol):
    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode document text for storage."""

    async def embed_query(self, text: str) -> list[float]:
        """Encode one retrieval query."""


def _load_beir_runtime() -> tuple[Any, Any, Any]:
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError as exc:
        raise RuntimeError(
            "knowledge-beir-eval requires optional dependencies. "
            "Install them with `uv sync --extra beir_eval --extra web_knowledge` or equivalent."
        ) from exc
    return util, GenericDataLoader, EvaluateRetrieval


def _default_bench_dir(workspace: Path, dataset: str) -> Path:
    return workspace / "benchmarks" / "beir" / dataset


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _doc_url(dataset: str, doc_id: str) -> str:
    return f"beir://{dataset}/{quote(doc_id, safe='')}"


def _doc_id_from_url(url: str, dataset: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme != "beir" or parsed.netloc != dataset:
        return None
    path = parsed.path.lstrip("/")
    if not path:
        return None
    return unquote(path)


def _score_rank(rank: int) -> float:
    return 1.0 / float(rank)


def _build_results_payload(
    candidate_parents: list[dict[str, Any]],
    *,
    dataset: str,
) -> tuple[dict[str, float], list[str]]:
    ranked_doc_ids: list[str] = []
    scores: dict[str, float] = {}
    for parent in candidate_parents:
        doc_id = _doc_id_from_url(str(parent.get("url") or ""), dataset)
        if not doc_id or doc_id in scores:
            continue
        ranked_doc_ids.append(doc_id)
        scores[doc_id] = _score_rank(len(ranked_doc_ids))
    return scores, ranked_doc_ids


def _normalize_metric_keys(metrics: dict[Any, Any]) -> dict[int, float]:
    normalized: dict[int, float] = {}
    for key, value in metrics.items():
        if isinstance(key, int):
            normalized[int(key)] = float(value)
            continue
        if isinstance(key, str):
            match = re.search(r"@(\d+)$", key.strip())
            if match:
                normalized[int(match.group(1))] = float(value)
                continue
        raise ValueError(f"Unsupported BEIR metric key format: {key!r}")
    return normalized


def _query_case_metrics(gold_doc_ids: list[str], retrieved_doc_ids: list[str], evidence_doc_ids: list[str]) -> dict[str, Any]:
    gold = set(gold_doc_ids)
    retrieved = set(retrieved_doc_ids)
    evidence = set(evidence_doc_ids)
    relevant_retrieved = gold & retrieved
    relevant_in_evidence = gold & evidence
    return {
        "support_available_in_candidates": bool(relevant_retrieved),
        "support_available_in_evidence": bool(relevant_in_evidence),
        "relevant_retrieved_count": len(relevant_retrieved),
        "relevant_evidence_count": len(relevant_in_evidence),
        "recall_at_doc_limit": (len(relevant_retrieved) / len(gold)) if gold else 1.0,
    }


def _aggregate_case_diagnostics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {
            "average_candidate_parents": 0.0,
            "average_evidence_children": 0.0,
            "relevant_doc_in_candidate_parents_rate": 0.0,
            "relevant_doc_in_evidence_rate": 0.0,
        }

    count = len(cases)
    return {
        "average_candidate_parents": sum(len(case["top_candidate_parent_ids"]) for case in cases) / count,
        "average_evidence_children": sum(len(case["top_evidence_child_ids"]) for case in cases) / count,
        "relevant_doc_in_candidate_parents_rate": (
            sum(int(case["metrics"]["support_available_in_candidates"]) for case in cases) / count
        ),
        "relevant_doc_in_evidence_rate": (
            sum(int(case["metrics"]["support_available_in_evidence"]) for case in cases) / count
        ),
    }


def _aggregate_latency(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {
            "average_query_search_latency_seconds": 0.0,
            "slowest_queries": [],
        }

    slowest_queries = sorted(
        (
            {
                "query_id": str(case["query_id"]),
                "query": str(case["query"]),
                "query_search_latency_seconds": float(case["query_search_latency_seconds"]),
            }
            for case in cases
        ),
        key=lambda item: (
            -float(item["query_search_latency_seconds"]),
            str(item["query_id"]),
        ),
    )[:_SLOWEST_QUERY_COUNT]

    return {
        "average_query_search_latency_seconds": (
            sum(float(case["query_search_latency_seconds"]) for case in cases) / len(cases)
        ),
        "slowest_queries": slowest_queries,
    }


async def _run_knowledge_beir_eval_async(
    *,
    workspace: Path,
    provider: Any,
    model: str,
    dataset: str,
    bench_dir: Path | None,
    embedding_model: str,
    doc_limit: int,
    evidence_limit: int,
    service: WebKnowledgeService | None,
    db: WebKnowledgeDatabase | None,
    embedder: EmbeddingBackend | None,
    api_key: str | None,
) -> dict[str, Any]:
    if dataset not in _SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported BEIR dataset: {dataset}")
    if doc_limit != DEFAULT_DOC_LIMIT:
        raise ValueError(f"knowledge-beir-eval requires doc_limit={DEFAULT_DOC_LIMIT}")
    if evidence_limit != DEFAULT_EVIDENCE_LIMIT:
        raise ValueError(f"knowledge-beir-eval requires evidence_limit={DEFAULT_EVIDENCE_LIMIT}")

    util, generic_data_loader_cls, evaluate_retrieval_cls = _load_beir_runtime()

    resolved_bench_dir = _ensure_dir((bench_dir or _default_bench_dir(workspace, dataset)).expanduser())
    datasets_dir = _ensure_dir(resolved_bench_dir / "datasets")
    knowledge_workspace = _ensure_dir(resolved_bench_dir / "workspace")

    if service is None:
        config = KnowledgeConfig(
            enabled=True,
            embedding={"model": embedding_model},
            doc_limit=doc_limit,
            evidence_limit=evidence_limit,
        )
        service = WebKnowledgeService(
            workspace=knowledge_workspace,
            config=config,
            api_key=api_key,
            db=db,
            embedder=embedder,
        )

    dataset_path = Path(
        util.download_and_unzip(_DEFAULT_DATASET_URLS[dataset], str(datasets_dir))
    ).expanduser()
    corpus, queries, qrels = generic_data_loader_cls(data_folder=str(dataset_path)).load(split="test")

    ingestion_summary = {"inserted": 0, "updated": 0, "unchanged": 0, "failed": 0}
    for doc_id, doc in corpus.items():
        title = str(doc.get("title") or "").strip()
        raw_text = str(doc.get("text") or "").strip()
        try:
            result = await service.ingest_document(
                source_url=_doc_url(dataset, doc_id),
                final_url=_doc_url(dataset, doc_id),
                title=title,
                raw_text=raw_text,
                extractor="beir",
                status=200,
                is_partial=False,
            )
        except Exception:
            ingestion_summary["failed"] += 1
            continue

        status = str((result or {}).get("status") or "").strip().lower()
        if status in ingestion_summary:
            ingestion_summary[status] += 1
        else:
            ingestion_summary["failed"] += 1

    results: dict[str, dict[str, float]] = {}
    cases: list[dict[str, Any]] = []
    for qid, query_text in queries.items():
        query_search_started_at = time.perf_counter()
        search_result = await service.search_local(str(query_text))
        candidate_parents = search_result.parents[:doc_limit]
        _, evidence_children = select_evidence(
            search_result, service.config, doc_limit=doc_limit, evidence_limit=evidence_limit
        )
        query_search_latency_seconds = time.perf_counter() - query_search_started_at
        result_scores, retrieved_doc_ids = _build_results_payload(
            [parent.to_dict() for parent in candidate_parents],
            dataset=dataset,
        )
        results[str(qid)] = result_scores

        evidence_doc_ids: list[str] = []
        top_evidence_parent_ids: list[int] = []
        top_evidence_child_ids: list[int] = []
        for chunk in evidence_children:
            top_evidence_parent_ids.append(chunk.parent_id)
            top_evidence_child_ids.append(chunk.child_id)
            doc_id = _doc_id_from_url(chunk.url, dataset)
            if doc_id and doc_id not in evidence_doc_ids:
                evidence_doc_ids.append(doc_id)

        gold_doc_ids = sorted(
            str(doc_id)
            for doc_id, relevance in dict(qrels.get(qid, {})).items()
            if int(relevance) > 0
        )
        case_metrics = _query_case_metrics(gold_doc_ids, retrieved_doc_ids, evidence_doc_ids)
        cases.append(
            {
                "query_id": str(qid),
                "query": str(query_text),
                "gold_doc_ids": gold_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "top_candidate_parent_ids": [
                    parent.parent_id for parent in candidate_parents
                ],
                "top_evidence_parent_ids": top_evidence_parent_ids,
                "top_evidence_child_ids": top_evidence_child_ids,
                "evidence_doc_ids": evidence_doc_ids,
                "query_search_latency_seconds": query_search_latency_seconds,
                "metrics": case_metrics,
            }
        )

    evaluator = evaluate_retrieval_cls()
    ndcg, map_scores, recall, precision = evaluator.evaluate(qrels, results, _METRIC_CUTOFFS)
    mrr = evaluator.evaluate_custom(qrels, results, _MRR_CUTOFFS, metric="mrr")

    return {
        "report_version": REPORT_VERSION,
        "evaluation": "knowledge_beir",
        "dataset": dataset,
        "bench_dir": str(resolved_bench_dir),
        "dataset_path": str(dataset_path),
        "model": model,
        "embedding_model": embedding_model,
        "doc_limit": doc_limit,
        "evidence_limit": evidence_limit,
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "ingestion_summary": ingestion_summary,
        "summary": {
            "ndcg": _normalize_metric_keys(ndcg),
            "map": _normalize_metric_keys(map_scores),
            "recall": _normalize_metric_keys(recall),
            "precision": _normalize_metric_keys(precision),
            "mrr": _normalize_metric_keys(mrr),
        },
        "diagnostics": _aggregate_case_diagnostics(cases),
        "latency": _aggregate_latency(cases),
        "cases": cases,
    }


def run_knowledge_beir_eval(
    *,
    workspace: Path,
    provider: Any,
    model: str,
    dataset: str = DEFAULT_DATASET,
    bench_dir: Path | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    doc_limit: int = DEFAULT_DOC_LIMIT,
    evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
    service: WebKnowledgeService | None = None,
    db: WebKnowledgeDatabase | None = None,
    embedder: EmbeddingBackend | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        _run_knowledge_beir_eval_async(
            workspace=workspace,
            provider=provider,
            model=model,
            dataset=dataset,
            bench_dir=bench_dir,
            embedding_model=embedding_model,
            doc_limit=doc_limit,
            evidence_limit=evidence_limit,
            service=service,
            db=db,
            embedder=embedder,
            api_key=api_key,
        )
    )


def render_knowledge_beir_report_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    diagnostics = report["diagnostics"]
    latency = report["latency"]
    lines = [
        "# External Knowledge BEIR Report",
        "",
        "This report measures the current production parent-child retrieval path.",
        (
            f"Document recall is capped at top-{report['doc_limit']} because the evaluation "
            f"keeps `doc_limit <= {report['doc_limit']}` for this run."
        ),
        "These results use the same retrieval and rerank path as production, with no benchmark-only overrides.",
        "",
        "## Overview",
        f"- Dataset: `{report['dataset']}`",
        f"- Model: `{report['model']}`",
        f"- Embedding Model: `{report['embedding_model']}`",
        f"- Doc Limit: `{report['doc_limit']}`",
        f"- Evidence Limit: `{report['evidence_limit']}`",
        f"- Corpus Size: `{report['corpus_size']}`",
        f"- Query Count: `{report['query_count']}`",
        "",
        "## Metrics",
        "- nDCG: "
        + ", ".join(f"@{cutoff}=`{summary['ndcg'][cutoff]:.4f}`" for cutoff in _METRIC_CUTOFFS),
        "- MAP: "
        + ", ".join(f"@{cutoff}=`{summary['map'][cutoff]:.4f}`" for cutoff in _METRIC_CUTOFFS),
        "- Recall: "
        + ", ".join(f"@{cutoff}=`{summary['recall'][cutoff]:.4f}`" for cutoff in _METRIC_CUTOFFS),
        "- Precision: "
        + ", ".join(f"@{cutoff}=`{summary['precision'][cutoff]:.4f}`" for cutoff in _METRIC_CUTOFFS),
        "- MRR: "
        + ", ".join(f"@{cutoff}=`{summary['mrr'][cutoff]:.4f}`" for cutoff in _MRR_CUTOFFS),
        "",
        "## Diagnostics",
        f"- Average candidate parents: `{diagnostics['average_candidate_parents']:.3f}`",
        f"- Average evidence children: `{diagnostics['average_evidence_children']:.3f}`",
        (
            "- Relevant doc in candidate parents rate: "
            f"`{diagnostics['relevant_doc_in_candidate_parents_rate']:.3f}`"
        ),
        f"- Relevant doc in evidence rate: `{diagnostics['relevant_doc_in_evidence_rate']:.3f}`",
        "",
        "## Latency",
        (
            "- Average query search latency: "
            f"`{latency['average_query_search_latency_seconds']:.3f}` seconds"
        ),
        "- Slowest queries:",
        *[
            (
                f"- `{item['query_id']}` `{float(item['query_search_latency_seconds']):.3f}`s "
                f"{item['query']}"
            )
            for item in latency["slowest_queries"]
        ],
        "",
        "## Ingestion",
        f"- Inserted: `{report['ingestion_summary']['inserted']}`",
        f"- Updated: `{report['ingestion_summary']['updated']}`",
        f"- Unchanged: `{report['ingestion_summary']['unchanged']}`",
        f"- Failed: `{report['ingestion_summary']['failed']}`",
    ]
    return "\n".join(lines)


def save_knowledge_beir_report(report: dict[str, Any], save_dir: Path) -> dict[str, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "knowledge_beir_eval_report.json"
    markdown_path = save_dir / "knowledge_beir_eval_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_knowledge_beir_report_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
