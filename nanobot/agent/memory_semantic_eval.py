"""Semantic evaluation runner for v2 structured memory extraction."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from nanobot.agent.memory_db import MemoryDatabase, MemoryRecord, MemorySnapshot
from nanobot.agent.memory_service import MemoryService
from nanobot.agent.memory_sync import MemorySynchronizer

REPORT_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EXTRACT_THRESHOLD = 0.80
DEFAULT_OVERWRITE_THRESHOLD = 0.88
DEFAULT_JUDGE_CANDIDATE_FLOOR = 0.55
DEFAULT_JUDGE_TOP_K = 3

_LEVELS = {"L0", "L1"}
_SCENARIOS = {"extraction", "overwrite_preservation"}
_DEFAULT_FIXTURE_ROOT = Path("nanobot/tests/phase_7/fixtures/memory_eval_v2")


class EmbeddingBackend(Protocol):
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into normalized embedding vectors."""


class _SentenceTransformerBackend:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "memory-v2-eval requires optional dependencies. "
                "Install them with `uv sync --extra memory_eval` or equivalent."
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [[float(value) for value in row.tolist()] for row in embeddings]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_cases_root() -> Path:
    return _repo_root() / _DEFAULT_FIXTURE_ROOT


def _load_case_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("cases"), list):
            return [dict(item) for item in payload["cases"] if isinstance(item, dict)]
        if payload.get("case_id"):
            return [dict(payload)]
    raise ValueError(f"Unsupported semantic eval fixture format: {path}")


def _normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ValueError("Conversation messages must be objects")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Conversation messages must include non-empty string content")
    role = str(message.get("role") or "").strip()
    if not role:
        raise ValueError("Conversation messages must include role")
    normalized = dict(message)
    normalized["role"] = role
    normalized["content"] = content
    return normalized


def _normalize_snapshot(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        main_class = str(item.get("main_class") or "").strip()
        sub_class = " ".join(str(item.get("sub_class") or "").split())
        text = " ".join(str(item.get("text") or "").split())
        if not main_class or not sub_class or not text:
            continue
        key = (main_class, sub_class, text)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "main_class": main_class,
            "sub_class": sub_class,
            "text": text,
        })

    return sorted(
        normalized,
        key=lambda item: (item["main_class"], item["sub_class"], item["text"]),
    )


def _normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("Semantic eval cases must include case_id")

    level = str(case.get("level") or "").strip().upper()
    if level not in _LEVELS:
        raise ValueError(f"Case {case_id} has unsupported level: {level or '(empty)'}")

    scenario = str(case.get("scenario") or "").strip().lower()
    if scenario not in _SCENARIOS:
        raise ValueError(f"Case {case_id} has unsupported scenario: {scenario or '(empty)'}")

    conversation = case.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError(f"Case {case_id} must include a non-empty conversation list")

    return {
        **case,
        "case_id": case_id,
        "level": level,
        "scenario": scenario,
        "conversation": [_normalize_message(message) for message in conversation],
        "prior_snapshot": _normalize_snapshot(case.get("prior_snapshot") or []),
        "gold_final_snapshot": _normalize_snapshot(case.get("gold_final_snapshot") or []),
        "gold_extracted_snapshot": _normalize_snapshot(case.get("gold_extracted_snapshot") or []),
        "gold_preserved_snapshot": _normalize_snapshot(case.get("gold_preserved_snapshot") or []),
        "notes": str(case.get("notes") or "").strip(),
    }


def load_memory_semantic_cases(
    cases_path: Path | None = None,
    *,
    level: str = "all",
    scenario: str = "all",
) -> list[dict[str, Any]]:
    level_key = level.strip().upper()
    if level_key not in {"ALL", *sorted(_LEVELS)}:
        raise ValueError(f"Unsupported level filter: {level}")

    scenario_key = scenario.strip().lower()
    if scenario_key not in {"all", *_SCENARIOS}:
        raise ValueError(f"Unsupported scenario filter: {scenario}")

    source = cases_path.resolve() if cases_path else _default_cases_root()
    if not source.exists():
        raise FileNotFoundError(f"Semantic eval fixtures not found: {source}")

    raw_cases: list[dict[str, Any]] = []
    if source.is_file():
        raw_cases.extend(_load_case_payload(source))
    else:
        for path in sorted(source.rglob("*.json")):
            raw_cases.extend(_load_case_payload(path))

    normalized_cases = [_normalize_case(case) for case in raw_cases]
    filtered = [
        case
        for case in normalized_cases
        if (level_key == "ALL" or case["level"] == level_key)
        and (scenario_key == "all" or case["scenario"] == scenario_key)
    ]
    if not filtered:
        raise ValueError("No semantic eval cases matched the requested filters")
    return filtered


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, (2 * precision * recall) / (precision + recall)


def _render_embedding_text(item: dict[str, str]) -> str:
    return item["text"]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _load_hungarian() -> Any:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError(
            "memory-v2-eval requires optional dependencies. "
            "Install them with `uv sync --extra memory_eval` or equivalent."
        ) from exc
    return linear_sum_assignment


def _match_bucket(
    predicted_items: list[dict[str, str]],
    gold_items: list[dict[str, str]],
    *,
    threshold: float,
    embedder: EmbeddingBackend,
) -> dict[str, Any]:
    if not predicted_items and not gold_items:
        return {
            "accepted_matches": [],
            "unmatched_predicted": [],
            "unmatched_gold": [],
        }
    if not predicted_items:
        return {
            "accepted_matches": [],
            "unmatched_predicted": [],
            "unmatched_gold": list(gold_items),
        }
    if not gold_items:
        return {
            "accepted_matches": [],
            "unmatched_predicted": list(predicted_items),
            "unmatched_gold": [],
        }

    predicted_vectors = embedder.encode_texts([_render_embedding_text(item) for item in predicted_items])
    gold_vectors = embedder.encode_texts([_render_embedding_text(item) for item in gold_items])
    similarity_matrix = [
        [_cosine_similarity(predicted_vector, gold_vector) for gold_vector in gold_vectors]
        for predicted_vector in predicted_vectors
    ]

    linear_sum_assignment = _load_hungarian()
    row_indexes, col_indexes = linear_sum_assignment(
        [[1.0 - score for score in row] for row in similarity_matrix]
    )

    matched_predicted: set[int] = set()
    matched_gold: set[int] = set()
    accepted_matches: list[dict[str, Any]] = []

    for row_index, col_index in zip(list(row_indexes), list(col_indexes)):
        similarity = float(similarity_matrix[int(row_index)][int(col_index)])
        if similarity < threshold:
            continue
        matched_predicted.add(int(row_index))
        matched_gold.add(int(col_index))
        accepted_matches.append({
            "main_class": predicted_items[int(row_index)]["main_class"],
            "predicted": predicted_items[int(row_index)],
            "gold": gold_items[int(col_index)],
            "similarity": similarity,
            "match_stage": "semantic",
        })

    accepted_matches.sort(
        key=lambda item: (
            item["main_class"],
            item["gold"]["sub_class"],
            item["gold"]["text"],
            item["predicted"]["sub_class"],
            item["predicted"]["text"],
        )
    )
    return {
        "accepted_matches": accepted_matches,
        "unmatched_predicted": [
            item for index, item in enumerate(predicted_items) if index not in matched_predicted
        ],
        "unmatched_gold": [item for index, item in enumerate(gold_items) if index not in matched_gold],
    }


def _match_snapshots(
    predicted_snapshot: list[dict[str, str]],
    gold_snapshot: list[dict[str, str]],
    *,
    threshold: float,
    embedder: EmbeddingBackend,
) -> dict[str, Any]:
    predicted_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    gold_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)

    for item in predicted_snapshot:
        predicted_by_class[item["main_class"]].append(item)
    for item in gold_snapshot:
        gold_by_class[item["main_class"]].append(item)

    accepted_matches: list[dict[str, Any]] = []
    unmatched_predicted: list[dict[str, str]] = []
    unmatched_gold: list[dict[str, str]] = []

    for main_class in sorted(set(predicted_by_class) | set(gold_by_class)):
        bucket = _match_bucket(
            sorted(
                predicted_by_class.get(main_class, []),
                key=lambda item: (item["sub_class"], item["text"]),
            ),
            sorted(
                gold_by_class.get(main_class, []),
                key=lambda item: (item["sub_class"], item["text"]),
            ),
            threshold=threshold,
            embedder=embedder,
        )
        accepted_matches.extend(bucket["accepted_matches"])
        unmatched_predicted.extend(bucket["unmatched_predicted"])
        unmatched_gold.extend(bucket["unmatched_gold"])

    unmatched_predicted.sort(key=lambda item: (item["main_class"], item["sub_class"], item["text"]))
    unmatched_gold.sort(key=lambda item: (item["main_class"], item["sub_class"], item["text"]))
    return {
        "accepted_matches": accepted_matches,
        "unmatched_predicted": unmatched_predicted,
        "unmatched_gold": unmatched_gold,
    }


def _scenario_threshold(
    scenario: str,
    *,
    extract_threshold: float,
    overwrite_threshold: float,
) -> float:
    return overwrite_threshold if scenario == "overwrite_preservation" else extract_threshold


def _score_case(
    *,
    case: dict[str, Any],
    predicted_snapshot: list[dict[str, str]],
    threshold: float,
    embedder: EmbeddingBackend,
) -> dict[str, Any]:
    final_matching = _match_snapshots(
        predicted_snapshot,
        case["gold_final_snapshot"],
        threshold=threshold,
        embedder=embedder,
    )
    extracted_matching = _match_snapshots(
        predicted_snapshot,
        case["gold_extracted_snapshot"],
        threshold=threshold,
        embedder=embedder,
    )
    preserved_matching = _match_snapshots(
        predicted_snapshot,
        case["gold_preserved_snapshot"],
        threshold=threshold,
        embedder=embedder,
    )
    return _build_case_metrics(
        case=case,
        final_matching=final_matching,
        extracted_matching=extracted_matching,
        preserved_matching=preserved_matching,
    )


def _build_case_metrics(
    *,
    case: dict[str, Any],
    final_matching: dict[str, Any],
    extracted_matching: dict[str, Any],
    preserved_matching: dict[str, Any],
) -> dict[str, Any]:
    tp = len(final_matching["accepted_matches"])
    fp = len(final_matching["unmatched_predicted"])
    fn = len(final_matching["unmatched_gold"])
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

    extracted_gold_count = len(case["gold_extracted_snapshot"])
    preserved_gold_count = len(case["gold_preserved_snapshot"])
    extracted_match_count = len(extracted_matching["accepted_matches"])
    preserved_match_count = len(preserved_matching["accepted_matches"])

    return {
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": not final_matching["unmatched_predicted"] and not final_matching["unmatched_gold"],
        "extracted_gold_count": extracted_gold_count,
        "extracted_match_count": extracted_match_count,
        "preserved_gold_count": preserved_gold_count,
        "preserved_match_count": preserved_match_count,
        "extraction_recall": (
            extracted_match_count / extracted_gold_count if extracted_gold_count else None
        ),
        "preservation_recall": (
            preserved_match_count / preserved_gold_count if preserved_gold_count else None
        ),
        "accepted_matches": final_matching["accepted_matches"],
        "semantic_matches": final_matching.get("semantic_matches", final_matching["accepted_matches"]),
        "judge_matches": final_matching.get("judge_matches", []),
        "semantic_unmatched_gold": final_matching.get(
            "semantic_unmatched_gold",
            final_matching["unmatched_gold"],
        ),
        "semantic_unmatched_predicted": final_matching.get(
            "semantic_unmatched_predicted",
            final_matching["unmatched_predicted"],
        ),
        "unmatched_gold_items": final_matching["unmatched_gold"],
        "unmatched_predicted_items": final_matching["unmatched_predicted"],
        "missing_extracted_items": extracted_matching["unmatched_gold"],
        "dropped_preserved_items": preserved_matching["unmatched_gold"],
    }


def _sort_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        matches,
        key=lambda item: (
            item["main_class"],
            item["gold"]["sub_class"],
            item["gold"]["text"],
            item["predicted"]["sub_class"],
            item["predicted"]["text"],
        ),
    )


def _truncate_reason(reason: str, limit: int = 140) -> str:
    collapsed = " ".join(reason.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


async def _judge_memory_equivalence(
    *,
    provider: Any,
    model: str,
    scenario: str,
    main_class: str,
    predicted: dict[str, str],
    gold: dict[str, str],
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are judging whether two memory entries represent the same long-term memory. "
                "Return JSON only: "
                '{"verdict":"match"|"no_match","reason":"short explanation"}. '
                "Allow paraphrases, added scope clarifications, and wording expansions when the core fact is the same. "
                "Do not match entries with different values, entities, projects, preferences, states, or updated facts."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "scenario": scenario,
                    "main_class": main_class,
                    "predicted_memory": predicted,
                    "gold_memory": gold,
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = await provider.chat_with_retry(messages=messages, model=model, temperature=0.0)
    content = (response.content or "").strip()
    if not content:
        return {"verdict": "no_match", "reason": "empty judge response"}
    try:
        payload = json.loads(content)
        verdict = str(payload.get("verdict") or "").strip().lower()
        reason = str(payload.get("reason") or "").strip() or "no reason provided"
        if verdict not in {"match", "no_match"}:
            verdict = "no_match"
        return {"verdict": verdict, "reason": reason}
    except Exception:
        lowered = content.lower()
        verdict = "match" if "match" in lowered and "no_match" not in lowered else "no_match"
        return {"verdict": verdict, "reason": _truncate_reason(content)}


async def _judge_unmatched_pairs(
    *,
    unmatched_predicted: list[dict[str, str]],
    unmatched_gold: list[dict[str, str]],
    scenario: str,
    embedder: EmbeddingBackend,
    judge_provider: Any,
    judge_model: str,
    judge_cache: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    if not unmatched_predicted or not unmatched_gold:
        return {
            "judge_matches": [],
            "unmatched_predicted": list(unmatched_predicted),
            "unmatched_gold": list(unmatched_gold),
        }

    predicted_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    gold_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in unmatched_predicted:
        predicted_by_class[item["main_class"]].append(item)
    for item in unmatched_gold:
        gold_by_class[item["main_class"]].append(item)

    judge_matches: list[dict[str, Any]] = []
    remaining_predicted_all: list[dict[str, str]] = []
    remaining_gold_all: list[dict[str, str]] = []

    for main_class in sorted(set(predicted_by_class) | set(gold_by_class)):
        predicted_items = sorted(
            predicted_by_class.get(main_class, []),
            key=lambda item: (item["sub_class"], item["text"]),
        )
        gold_items = sorted(
            gold_by_class.get(main_class, []),
            key=lambda item: (item["sub_class"], item["text"]),
        )
        if not predicted_items or not gold_items:
            remaining_predicted_all.extend(predicted_items)
            remaining_gold_all.extend(gold_items)
            continue

        predicted_vectors = embedder.encode_texts([_render_embedding_text(item) for item in predicted_items])
        gold_vectors = embedder.encode_texts([_render_embedding_text(item) for item in gold_items])
        similarity_matrix = [
            [_cosine_similarity(predicted_vector, gold_vector) for gold_vector in gold_vectors]
            for predicted_vector in predicted_vectors
        ]

        candidate_pairs: list[tuple[int, int, float]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for predicted_index, scores in enumerate(similarity_matrix):
            ranked = sorted(
                enumerate(scores),
                key=lambda item: item[1],
                reverse=True,
            )
            for gold_index, similarity in ranked[:DEFAULT_JUDGE_TOP_K]:
                if similarity < DEFAULT_JUDGE_CANDIDATE_FLOOR:
                    continue
                key = (predicted_index, gold_index)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                candidate_pairs.append((predicted_index, gold_index, float(similarity)))

        positive_pairs: list[dict[str, Any]] = []
        for predicted_index, gold_index, similarity in candidate_pairs:
            predicted = predicted_items[predicted_index]
            gold = gold_items[gold_index]
            cache_key = (scenario, main_class, predicted["text"], gold["text"])
            if cache_key not in judge_cache:
                judge_cache[cache_key] = await _judge_memory_equivalence(
                    provider=judge_provider,
                    model=judge_model,
                    scenario=scenario,
                    main_class=main_class,
                    predicted=predicted,
                    gold=gold,
                )
            verdict = judge_cache[cache_key]
            if verdict["verdict"] != "match":
                continue
            positive_pairs.append({
                "main_class": main_class,
                "predicted": predicted,
                "gold": gold,
                "similarity": similarity,
                "judge_verdict": verdict["verdict"],
                "judge_reason_short": _truncate_reason(str(verdict.get("reason") or "")),
                "match_stage": "judge",
            })

        positive_pairs.sort(
            key=lambda item: (
                -item["similarity"],
                item["predicted"]["sub_class"],
                item["predicted"]["text"],
                item["gold"]["sub_class"],
                item["gold"]["text"],
            )
        )
        matched_predicted_keys: set[tuple[str, str, str]] = set()
        matched_gold_keys: set[tuple[str, str, str]] = set()
        bucket_matches: list[dict[str, Any]] = []

        for pair in positive_pairs:
            predicted_key = (
                pair["predicted"]["main_class"],
                pair["predicted"]["sub_class"],
                pair["predicted"]["text"],
            )
            gold_key = (
                pair["gold"]["main_class"],
                pair["gold"]["sub_class"],
                pair["gold"]["text"],
            )
            if predicted_key in matched_predicted_keys or gold_key in matched_gold_keys:
                continue
            matched_predicted_keys.add(predicted_key)
            matched_gold_keys.add(gold_key)
            bucket_matches.append(pair)

        judge_matches.extend(bucket_matches)
        remaining_predicted_all.extend([
            item
            for item in predicted_items
            if (item["main_class"], item["sub_class"], item["text"]) not in matched_predicted_keys
        ])
        remaining_gold_all.extend([
            item
            for item in gold_items
            if (item["main_class"], item["sub_class"], item["text"]) not in matched_gold_keys
        ])

    return {
        "judge_matches": _sort_matches(judge_matches),
        "unmatched_predicted": sorted(
            remaining_predicted_all,
            key=lambda item: (item["main_class"], item["sub_class"], item["text"]),
        ),
        "unmatched_gold": sorted(
            remaining_gold_all,
            key=lambda item: (item["main_class"], item["sub_class"], item["text"]),
        ),
    }


async def _hybrid_match_snapshots(
    *,
    predicted_snapshot: list[dict[str, str]],
    gold_snapshot: list[dict[str, str]],
    scenario: str,
    threshold: float,
    embedder: EmbeddingBackend,
    judge_provider: Any,
    judge_model: str,
    judge_cache: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    semantic_matching = _match_snapshots(
        predicted_snapshot,
        gold_snapshot,
        threshold=threshold,
        embedder=embedder,
    )
    judge_matching = await _judge_unmatched_pairs(
        unmatched_predicted=semantic_matching["unmatched_predicted"],
        unmatched_gold=semantic_matching["unmatched_gold"],
        scenario=scenario,
        embedder=embedder,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_cache=judge_cache,
    )
    accepted_matches = _sort_matches(
        [*semantic_matching["accepted_matches"], *judge_matching["judge_matches"]]
    )
    return {
        "accepted_matches": accepted_matches,
        "semantic_matches": semantic_matching["accepted_matches"],
        "judge_matches": judge_matching["judge_matches"],
        "semantic_unmatched_predicted": semantic_matching["unmatched_predicted"],
        "semantic_unmatched_gold": semantic_matching["unmatched_gold"],
        "unmatched_predicted": judge_matching["unmatched_predicted"],
        "unmatched_gold": judge_matching["unmatched_gold"],
    }


async def _score_case_hybrid(
    *,
    case: dict[str, Any],
    predicted_snapshot: list[dict[str, str]],
    threshold: float,
    embedder: EmbeddingBackend,
    judge_provider: Any,
    judge_model: str,
) -> dict[str, Any]:
    judge_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    final_matching = await _hybrid_match_snapshots(
        predicted_snapshot=predicted_snapshot,
        gold_snapshot=case["gold_final_snapshot"],
        scenario=case["scenario"],
        threshold=threshold,
        embedder=embedder,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_cache=judge_cache,
    )
    extracted_matching = await _hybrid_match_snapshots(
        predicted_snapshot=predicted_snapshot,
        gold_snapshot=case["gold_extracted_snapshot"],
        scenario=case["scenario"],
        threshold=threshold,
        embedder=embedder,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_cache=judge_cache,
    )
    preserved_matching = await _hybrid_match_snapshots(
        predicted_snapshot=predicted_snapshot,
        gold_snapshot=case["gold_preserved_snapshot"],
        scenario=case["scenario"],
        threshold=threshold,
        embedder=embedder,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_cache=judge_cache,
    )
    return _build_case_metrics(
        case=case,
        final_matching=final_matching,
        extracted_matching=extracted_matching,
        preserved_matching=preserved_matching,
    )


def _aggregate_group(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    tp_total = fp_total = fn_total = 0
    exact_total = 0
    extracted_gold_total = extracted_match_total = 0
    preserved_gold_total = preserved_match_total = 0
    consolidate_successes = 0
    raw_archives = 0

    for result in case_results:
        metrics = result["metrics"]
        tp_total += metrics["true_positive_count"]
        fp_total += metrics["false_positive_count"]
        fn_total += metrics["false_negative_count"]
        exact_total += int(metrics["exact_match"])
        extracted_gold_total += metrics["extracted_gold_count"]
        extracted_match_total += metrics["extracted_match_count"]
        preserved_gold_total += metrics["preserved_gold_count"]
        preserved_match_total += metrics["preserved_match_count"]
        consolidate_successes += int(result["consolidate_returned_true"])
        raw_archives += int(result["raw_archive_detected"])

    precision, recall, f1 = _precision_recall_f1(tp_total, fp_total, fn_total)
    count = len(case_results)
    return {
        "case_count": count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": (exact_total / count) if count else 0.0,
        "extraction_recall": (
            extracted_match_total / extracted_gold_total if extracted_gold_total else None
        ),
        "preservation_recall": (
            preserved_match_total / preserved_gold_total if preserved_gold_total else None
        ),
        "consolidate_success_rate": (consolidate_successes / count) if count else 0.0,
        "raw_archive_rate": (raw_archives / count) if count else 0.0,
        "true_positive_count": tp_total,
        "false_positive_count": fp_total,
        "false_negative_count": fn_total,
        "extracted_gold_count": extracted_gold_total,
        "preserved_gold_count": preserved_gold_total,
    }


def _inject_prior_snapshot(workspace: Path, snapshot: list[dict[str, str]]) -> None:
    if not snapshot:
        return
    db = MemoryDatabase(workspace)
    db.initialize()
    records = tuple(
        MemoryRecord.create(item["main_class"], item["sub_class"], item["text"])
        for item in snapshot
    )
    db.commit_snapshot(
        MemorySnapshot(0, records), expected_revision=0, event_id="eval-seed",
        ts="1970-01-01T00:00:00", session_key="eval", history_text="",
        candidate_type="fixture",
    )
    MemorySynchronizer(db).sync()


async def _run_semantic_case(
    *,
    case: dict[str, Any],
    provider: Any,
    model: str,
) -> dict[str, Any]:
    case_id = case["case_id"]
    with TemporaryDirectory(prefix=f"nanobot-memory-semantic-{case_id}-") as tmp:
        workspace = Path(tmp)
        _inject_prior_snapshot(workspace, case["prior_snapshot"])

        service = MemoryService(workspace, provider, model)
        result = await service.consolidate(case["conversation"], session_key="eval")
        db = service.database
        memory_text = db.memory_file.read_text(encoding="utf-8") if db.memory_file.exists() else ""
        history_text = db.history_file.read_text(encoding="utf-8") if db.history_file.exists() else ""
        predicted_snapshot = _normalize_snapshot(db.list_canonical_memories())

        return {
            "consolidate_returned_true": result.database_committed,
            "raw_archive_detected": "[RAW]" in history_text,
            "predicted_snapshot": predicted_snapshot,
            "rendered_memory_markdown": memory_text,
            "rendered_history_markdown": history_text,
        }


async def _run_memory_v2_semantic_eval_async(
    cases: list[dict[str, Any]],
    *,
    provider: Any | None,
    provider_factory: Any | None,
    judge_provider: Any | None,
    judge_provider_factory: Any | None,
    model: str,
    judge_model: str,
    cases_source: str,
    embedding_model: str,
    extract_threshold: float,
    overwrite_threshold: float,
    embedder: EmbeddingBackend,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("At least one semantic memory evaluation case is required")
    if (provider is None) == (provider_factory is None):
        raise ValueError("Provide exactly one of provider or provider_factory")
    if judge_provider is not None and judge_provider_factory is not None:
        raise ValueError("Provide either judge_provider or judge_provider_factory, not both")

    case_results: list[dict[str, Any]] = []
    for case in cases:
        scenario_threshold = _scenario_threshold(
            case["scenario"],
            extract_threshold=extract_threshold,
            overwrite_threshold=overwrite_threshold,
        )
        case_provider = provider_factory(case) if provider_factory is not None else provider
        case_judge_provider = (
            judge_provider_factory(case)
            if judge_provider_factory is not None
            else (judge_provider or case_provider)
        )
        case_run = await _run_semantic_case(
            case=case,
            provider=case_provider,
            model=model,
        )
        metrics = await _score_case_hybrid(
            case=case,
            predicted_snapshot=case_run["predicted_snapshot"],
            threshold=scenario_threshold,
            embedder=embedder,
            judge_provider=case_judge_provider,
            judge_model=judge_model,
        )
        case_results.append({
            "case_id": case["case_id"],
            "level": case["level"],
            "scenario": case["scenario"],
            "notes": case["notes"],
            "conversation_message_count": len(case["conversation"]),
            "threshold": scenario_threshold,
            "prior_snapshot": case["prior_snapshot"],
            "gold_final_snapshot": case["gold_final_snapshot"],
            "gold_extracted_snapshot": case["gold_extracted_snapshot"],
            "gold_preserved_snapshot": case["gold_preserved_snapshot"],
            **case_run,
            "metrics": metrics,
        })

    by_level = {
        level: _aggregate_group([result for result in case_results if result["level"] == level])
        for level in sorted(_LEVELS)
        if any(result["level"] == level for result in case_results)
    }
    by_scenario = {
        scenario: _aggregate_group([result for result in case_results if result["scenario"] == scenario])
        for scenario in sorted(_SCENARIOS)
        if any(result["scenario"] == scenario for result in case_results)
    }
    by_level_scenario = {
        f"{level}/{scenario}": _aggregate_group(
            [
                result
                for result in case_results
                if result["level"] == level and result["scenario"] == scenario
            ]
        )
        for level in sorted(_LEVELS)
        for scenario in sorted(_SCENARIOS)
        if any(result["level"] == level and result["scenario"] == scenario for result in case_results)
    }

    return {
        "report_version": REPORT_VERSION,
        "evaluation": "memory_v2_semantic",
        "cases_source": cases_source,
        "mode": "v2",
        "model": model,
        "judge_model": judge_model,
        "embedding_model": embedding_model,
        "total_cases": len(case_results),
        "thresholds": {
            "extraction": extract_threshold,
            "overwrite_preservation": overwrite_threshold,
        },
        "judge_candidate_floor": DEFAULT_JUDGE_CANDIDATE_FLOOR,
        "judge_top_k": DEFAULT_JUDGE_TOP_K,
        "summary": {
            "overall": _aggregate_group(case_results),
            "by_level": by_level,
            "by_scenario": by_scenario,
            "by_level_scenario": by_level_scenario,
        },
        "cases": case_results,
    }


def run_memory_v2_semantic_eval(
    *,
    provider: Any | None = None,
    provider_factory: Any | None = None,
    judge_provider: Any | None = None,
    judge_provider_factory: Any | None = None,
    model: str,
    judge_model: str | None = None,
    cases: list[dict[str, Any]] | None = None,
    cases_path: Path | None = None,
    level: str = "all",
    scenario: str = "all",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    extract_threshold: float = DEFAULT_EXTRACT_THRESHOLD,
    overwrite_threshold: float = DEFAULT_OVERWRITE_THRESHOLD,
    embedder: EmbeddingBackend | None = None,
) -> dict[str, Any]:
    loaded_cases = (
        [_normalize_case(case) for case in cases]
        if cases is not None
        else load_memory_semantic_cases(
            cases_path,
            level=level,
            scenario=scenario,
        )
    )
    cases_source = str(cases_path.resolve()) if cases_path else str(_default_cases_root())
    resolved_embedder = embedder or _SentenceTransformerBackend(embedding_model)
    resolved_judge_model = judge_model.strip() if isinstance(judge_model, str) and judge_model.strip() else model
    return asyncio.run(
        _run_memory_v2_semantic_eval_async(
            loaded_cases,
            provider=provider,
            provider_factory=provider_factory,
            judge_provider=judge_provider,
            judge_provider_factory=judge_provider_factory,
            model=model,
            judge_model=resolved_judge_model,
            cases_source=cases_source,
            embedding_model=embedding_model,
            extract_threshold=extract_threshold,
            overwrite_threshold=overwrite_threshold,
            embedder=resolved_embedder,
        )
    )


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_memory_item(item: dict[str, Any]) -> str:
    return f"[{item['main_class']}/{item['sub_class']}] {item['text']}"


def render_memory_v2_semantic_report_markdown(report: dict[str, Any]) -> str:
    overall = report["summary"]["overall"]
    lines = [
        "# V2 Memory Semantic Evaluation Report",
        "",
        f"- Report Version: {report['report_version']}",
        f"- Cases Source: `{report['cases_source']}`",
        f"- Total Cases: {report['total_cases']}",
        f"- Mode: `{report['mode']}`",
        f"- Model: `{report['model']}`",
        f"- Judge Model: `{report['judge_model']}`",
        f"- Embedding Model: `{report['embedding_model']}`",
        f"- Extraction Threshold: `{report['thresholds']['extraction']:.2f}`",
        f"- Overwrite Threshold: `{report['thresholds']['overwrite_preservation']:.2f}`",
        f"- Judge Candidate Floor: `{report['judge_candidate_floor']:.2f}`",
        f"- Judge Top K: `{report['judge_top_k']}`",
        "",
        "## Overall",
        "",
        "| Precision | Recall | F1 | Exact Match | Extraction Recall | Preservation Recall | Success Rate | Raw Archive |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {_format_metric(overall['precision'])} | {_format_metric(overall['recall'])} | "
            f"{_format_metric(overall['f1'])} | {_format_metric(overall['exact_match_rate'])} | "
            f"{_format_metric(overall['extraction_recall'])} | {_format_metric(overall['preservation_recall'])} | "
            f"{_format_metric(overall['consolidate_success_rate'])} | {_format_metric(overall['raw_archive_rate'])} |"
        ),
        "",
        "## Breakdown",
        "",
        "| Group | Cases | Precision | Recall | F1 | Exact Match | Extraction Recall | Preservation Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for prefix, groups in (
        ("level", report["summary"]["by_level"]),
        ("scenario", report["summary"]["by_scenario"]),
        ("slice", report["summary"]["by_level_scenario"]),
    ):
        for name, metrics in groups.items():
            lines.append(
                f"| {prefix}:{name} | {metrics['case_count']} | {_format_metric(metrics['precision'])} | "
                f"{_format_metric(metrics['recall'])} | {_format_metric(metrics['f1'])} | "
                f"{_format_metric(metrics['exact_match_rate'])} | "
                f"{_format_metric(metrics['extraction_recall'])} | "
                f"{_format_metric(metrics['preservation_recall'])} |"
            )

    lines.extend([
        "",
        "## Case Results",
        "",
        "| Case | Level | Scenario | F1 | Semantic Matches | Judge Matches | Still Missing Gold | Still Extra Predicted | Dropped Preserved | Raw Archive |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])

    for case in report["cases"]:
        metrics = case["metrics"]
        lines.append(
            f"| {case['case_id']} | {case['level']} | {case['scenario']} | {_format_metric(metrics['f1'])} | "
            f"{len(metrics['semantic_matches'])} | {len(metrics['judge_matches'])} | "
            f"{len(metrics['unmatched_gold_items'])} | {len(metrics['unmatched_predicted_items'])} | "
            f"{len(metrics['dropped_preserved_items'])} | {case['raw_archive_detected']} |"
        )

    detail_cases = [
        case
        for case in report["cases"]
        if case["metrics"].get("semantic_unmatched_gold")
        or case["metrics"].get("semantic_unmatched_predicted")
    ]

    lines.extend([
        "",
        "## Case Details",
    ])

    if not detail_cases:
        lines.extend([
            "",
            "- (none)",
        ])

    for case in detail_cases:
        metrics = case["metrics"]
        lines.extend([
            "",
            f"### {case['case_id']}",
            "",
            f"- Level: `{case['level']}`",
            f"- Scenario: `{case['scenario']}`",
            f"- F1: `{_format_metric(metrics['f1'])}`",
            f"- Raw Archive: `{case['raw_archive_detected']}`",
            "- Semantic Matches:",
        ])
        if metrics["semantic_matches"]:
            lines.extend(
                [
                    "  - "
                    + f"{_format_memory_item(match['predicted'])} ~= {_format_memory_item(match['gold'])} "
                    + f"(sim={match['similarity']:.3f})"
                    for match in metrics["semantic_matches"]
                ]
            )
        else:
            lines.append("  - (none)")

        lines.append("- Judge Matches:")
        if metrics["judge_matches"]:
            lines.extend(
                [
                    "  - "
                    + f"{_format_memory_item(match['predicted'])} ~= {_format_memory_item(match['gold'])} "
                    + f"(sim={match['similarity']:.3f}; reason={match['judge_reason_short']})"
                    for match in metrics["judge_matches"]
                ]
            )
        else:
            lines.append("  - (none)")

        lines.append("- SBERT Unmatched Gold:")
        semantic_unmatched_gold = metrics.get("semantic_unmatched_gold", [])
        if semantic_unmatched_gold:
            lines.extend([f"  - {_format_memory_item(item)}" for item in semantic_unmatched_gold])
        else:
            lines.append("  - (none)")

        lines.append("- SBERT Unmatched Predicted:")
        semantic_unmatched_predicted = metrics.get("semantic_unmatched_predicted", [])
        if semantic_unmatched_predicted:
            lines.extend(
                [f"  - {_format_memory_item(item)}" for item in semantic_unmatched_predicted]
            )
        else:
            lines.append("  - (none)")

        lines.append("- Still Missing Gold:")
        if metrics["unmatched_gold_items"]:
            lines.extend([f"  - {_format_memory_item(item)}" for item in metrics["unmatched_gold_items"]])
        else:
            lines.append("  - (none)")

        lines.append("- Still Extra Predicted:")
        if metrics["unmatched_predicted_items"]:
            lines.extend(
                [f"  - {_format_memory_item(item)}" for item in metrics["unmatched_predicted_items"]]
            )
        else:
            lines.append("  - (none)")

    return "\n".join(lines)


def save_memory_v2_semantic_report(report: dict[str, Any], save_dir: Path) -> dict[str, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "memory_v2_semantic_eval_report.json"
    markdown_path = save_dir / "memory_v2_semantic_eval_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        render_memory_v2_semantic_report_markdown(report),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}
