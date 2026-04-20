# Knowledge BEIR eval test content:
# - verifies BEIR corpus documents are ingested into the external knowledge DB
# - verifies the runner uses the current search() limits and emits top-20 metrics only
# - verifies reruns dedupe unchanged documents and report parent-child diagnostics

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from nanobot.agent.knowledge import WebKnowledgeService
from nanobot.agent.knowledge_beir_eval import (
    _normalize_metric_keys,
    render_knowledge_beir_report_markdown,
    run_knowledge_beir_eval,
    save_knowledge_beir_report,
)
from nanobot.agent.knowledge_db import WebKnowledgeDatabase
from nanobot.config.schema import KnowledgeConfig
from nanobot.providers.base import LLMProvider, LLMResponse


class _KeywordEmbedder:
    dimension = 2

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            values = [
                1.0 if "alpha" in lowered else 0.0,
                1.0 if "shared" in lowered else 0.0,
            ]
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append([value / norm for value in values])
        return vectors


class _UnusedProvider(LLMProvider):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="")

    def get_default_model(self) -> str:
        return "test-model"


class _KeywordReranker:
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, text in pairs:
            query_lower = query.lower()
            text_lower = text.lower()
            score = 0.0
            if "alpha" in query_lower and "alpha" in text_lower:
                score += 10.0
            if "shared" in query_lower and "shared" in text_lower:
                score += 10.0
            scores.append(score)
        return scores


class _FakeBEIRUtil:
    @staticmethod
    def download_and_unzip(url: str, output_dir: str) -> str:
        path = Path(output_dir) / "scifact"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


class _FakeGenericDataLoader:
    def __init__(self, *, data_folder: str):
        self.data_folder = data_folder

    def load(self, split: str = "test"):
        assert split == "test"
        corpus = {
            "doc-alpha": {"title": "Alpha Claim", "text": "alpha evidence only"},
        }
        corpus.update(
            {
                f"doc-shared-{index:02d}": {
                    "title": f"Shared Claim {index:02d}",
                    "text": f"shared evidence item {index:02d}",
                }
                for index in range(21)
            }
        )
        queries = {
            "q1": "alpha",
            "q2": "shared",
        }
        qrels = {
            "q1": {"doc-alpha": 1},
            "q2": {f"doc-shared-{index:02d}": 1 for index in range(21)},
        }
        return corpus, queries, qrels


class _FakeEvaluateRetrieval:
    @staticmethod
    def _ranked_doc_ids(results: dict[str, dict[str, float]], qid: str) -> list[str]:
        return list(results.get(qid, {}).keys())

    def evaluate(
        self,
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
        ndcg: dict[int, float] = {}
        map_scores: dict[int, float] = {}
        recall: dict[int, float] = {}
        precision: dict[int, float] = {}
        for k in k_values:
            ndcg_values: list[float] = []
            map_values: list[float] = []
            recall_values: list[float] = []
            precision_values: list[float] = []
            for qid, qrel in qrels.items():
                gold = {doc_id for doc_id, relevance in qrel.items() if int(relevance) > 0}
                ranked = self._ranked_doc_ids(results, qid)[:k]
                hits = [1 if doc_id in gold else 0 for doc_id in ranked]
                hit_count = sum(hits)
                recall_values.append(hit_count / len(gold) if gold else 1.0)
                precision_values.append(hit_count / k if k else 1.0)

                dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(hits))
                ideal_hits = [1] * min(len(gold), k)
                idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal_hits))
                ndcg_values.append(dcg / idcg if idcg else 1.0)

                running_hits = 0
                ap = 0.0
                for index, rel in enumerate(hits, start=1):
                    if rel:
                        running_hits += 1
                        ap += running_hits / index
                map_values.append(ap / len(gold) if gold else 1.0)

            ndcg[k] = sum(ndcg_values) / len(ndcg_values)
            map_scores[k] = sum(map_values) / len(map_values)
            recall[k] = sum(recall_values) / len(recall_values)
            precision[k] = sum(precision_values) / len(precision_values)
        return ndcg, map_scores, recall, precision

    def evaluate_custom(
        self,
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        metric: str,
    ) -> dict[int, float]:
        assert metric == "mrr"
        scores: dict[int, float] = {}
        for k in k_values:
            values: list[float] = []
            for qid, qrel in qrels.items():
                gold = {doc_id for doc_id, relevance in qrel.items() if int(relevance) > 0}
                ranked = self._ranked_doc_ids(results, qid)[:k]
                reciprocal_rank = 0.0
                for index, doc_id in enumerate(ranked, start=1):
                    if doc_id in gold:
                        reciprocal_rank = 1.0 / index
                        break
                values.append(reciprocal_rank)
            scores[k] = sum(values) / len(values)
        return scores


def test_normalize_metric_keys_accepts_real_beir_style_keys() -> None:
    metrics = {
        "NDCG@1": 0.5,
        "NDCG@10": 0.7,
        "NDCG@100": 0.9,
    }

    assert _normalize_metric_keys(metrics) == {
        1: 0.5,
        10: 0.7,
        100: 0.9,
    }


def _service(tmp_path: Path) -> WebKnowledgeService:
    knowledge_workspace = tmp_path / "benchmarks" / "beir" / "scifact" / "workspace"
    return WebKnowledgeService(
        workspace=knowledge_workspace,
        provider=_UnusedProvider(),
        model="test-model",
        config=KnowledgeConfig(
            enabled=True,
            embedding_model="sentence-transformers/test-model",
            doc_limit=10,
            evidence_limit=5,
            chunk_chars=200,
            chunk_overlap_chars=20,
        ),
        db=WebKnowledgeDatabase(knowledge_workspace, vec_backend="array"),
        embedder=_KeywordEmbedder(),
        reranker=_KeywordReranker(),
    )


def test_run_knowledge_beir_eval_uses_current_search_limits_and_reports_current_cutoffs(monkeypatch, tmp_path: Path) -> None:
    from nanobot.agent import knowledge_beir_eval as beir_eval

    monkeypatch.setattr(
        beir_eval,
        "_load_beir_runtime",
        lambda: (_FakeBEIRUtil, _FakeGenericDataLoader, _FakeEvaluateRetrieval),
    )

    service = _service(tmp_path)
    recorded_limits: list[tuple[int | None, int | None]] = []
    original_search = service.search

    async def _recorded_search(query: str, *, doc_limit: int | None = None, evidence_limit: int | None = None):
        recorded_limits.append((doc_limit, evidence_limit))
        return await original_search(query, doc_limit=doc_limit, evidence_limit=evidence_limit)

    service.search = _recorded_search  # type: ignore[method-assign]

    report = run_knowledge_beir_eval(
        workspace=tmp_path,
        provider=_UnusedProvider(),
        model="test-model",
        dataset="scifact",
        bench_dir=tmp_path / "benchmarks" / "beir" / "scifact",
        service=service,
        embedding_model="sentence-transformers/test-model",
    )

    assert recorded_limits == [(10, 5), (10, 5)]
    assert report["evaluation"] == "knowledge_beir"
    assert report["doc_limit"] == 10
    assert report["evidence_limit"] == 5
    assert report["corpus_size"] == 22
    assert report["query_count"] == 2
    assert set(report["summary"]["ndcg"]) == {1, 3, 5, 10, 20, 50, 100}
    assert set(report["summary"]["mrr"]) == {10, 20, 50, 100}
    assert "summary_model" not in report
    assert report["diagnostics"]["average_candidate_parents"] <= 10.0
    assert report["diagnostics"]["average_evidence_children"] >= 0.0
    assert report["cases"][0]["top_candidate_parent_ids"]


def test_run_knowledge_beir_eval_rerun_marks_unchanged_docs(monkeypatch, tmp_path: Path) -> None:
    from nanobot.agent import knowledge_beir_eval as beir_eval

    monkeypatch.setattr(
        beir_eval,
        "_load_beir_runtime",
        lambda: (_FakeBEIRUtil, _FakeGenericDataLoader, _FakeEvaluateRetrieval),
    )

    service = _service(tmp_path)
    first = run_knowledge_beir_eval(
        workspace=tmp_path,
        provider=_UnusedProvider(),
        model="test-model",
        dataset="scifact",
        bench_dir=tmp_path / "benchmarks" / "beir" / "scifact",
        service=service,
        embedding_model="sentence-transformers/test-model",
    )
    second = run_knowledge_beir_eval(
        workspace=tmp_path,
        provider=_UnusedProvider(),
        model="test-model",
        dataset="scifact",
        bench_dir=tmp_path / "benchmarks" / "beir" / "scifact",
        service=service,
        embedding_model="sentence-transformers/test-model",
    )

    assert first["ingestion_summary"]["inserted"] == 22
    assert second["ingestion_summary"]["unchanged"] == 22
    assert second["summary"]["recall"][100] == second["summary"]["recall"][10]


def test_render_and_save_knowledge_beir_report(tmp_path: Path) -> None:
    report = {
        "report_version": 2,
        "evaluation": "knowledge_beir",
        "dataset": "scifact",
        "bench_dir": str(tmp_path / "bench"),
        "dataset_path": str(tmp_path / "bench" / "datasets" / "scifact"),
        "model": "test-model",
        "embedding_model": "fake-embedding-model",
        "doc_limit": 10,
        "evidence_limit": 5,
        "corpus_size": 22,
        "query_count": 2,
        "ingestion_summary": {"inserted": 22, "updated": 0, "unchanged": 0, "failed": 0},
        "summary": {
            "ndcg": {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0, 20: 0.95, 50: 0.92, 100: 0.9},
            "map": {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0, 20: 0.95, 50: 0.92, 100: 0.9},
            "recall": {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0, 20: 0.95, 50: 0.92, 100: 0.9},
            "precision": {1: 1.0, 3: 0.5, 5: 0.4, 10: 0.2, 20: 0.15, 50: 0.12, 100: 0.1},
            "mrr": {10: 1.0, 20: 1.0, 50: 1.0, 100: 1.0},
        },
        "diagnostics": {
            "average_candidate_parents": 10.5,
            "average_evidence_children": 2.5,
            "relevant_doc_in_candidate_parents_rate": 1.0,
            "relevant_doc_in_evidence_rate": 1.0,
        },
        "cases": [],
    }

    markdown = render_knowledge_beir_report_markdown(report)
    saved = save_knowledge_beir_report(report, tmp_path / "reports")

    assert "parent-child retrieval path" in markdown
    assert "top-10" in markdown
    assert "@1=`1.0000`" in markdown
    assert "@3=`1.0000`" in markdown
    assert "@5=`1.0000`" in markdown
    assert "@10=`1.0000`" in markdown
    assert "@20=`0.9500`" in markdown
    assert "@50=`0.9200`" in markdown
    assert "@100=`0.9000`" in markdown
    assert saved["json"].exists()
    assert saved["markdown"].exists()
