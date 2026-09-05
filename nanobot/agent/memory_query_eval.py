"""FTS retrieval + answer evaluation runner for v2 memory query benchmarks."""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory_db import MemoryDatabase, MemoryRecord, MemorySnapshot
from nanobot.agent.memory_service import MemoryService
from nanobot.agent.memory_sync import MemorySynchronizer

REPORT_VERSION = 1
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RETRIEVAL_THRESHOLD = 0.82
DEFAULT_TOP_K = 5
DEFAULT_MODE = "snapshot"

_LEVELS = {"L0", "L1"}
_QUESTION_TYPES = {
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
}
_MODES = {"snapshot", "replay"}
_CORE_CLASSES = {"personal_profile", "preferences", "constraints"}
_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"


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
                "memory-v2-query-eval requires optional dependencies. "
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








def _load_hungarian() -> Any:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError(
            "memory-v2-query-eval requires optional dependencies. "
            "Install them with `uv sync --extra memory_eval` or equivalent."
        ) from exc
    return linear_sum_assignment


def _load_case_payload(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("cases"), list):
            return [dict(item) for item in payload["cases"] if isinstance(item, dict)]
        if payload.get("case_id"):
            return [dict(payload)]
    raise ValueError(f"Unsupported query eval fixture format: {path}")


def _load_source_records(source_dataset_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(source_dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported LongMemEval source format: {source_dataset_path}")

    records: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id") or "").strip()
        if question_id:
            records[question_id] = dict(item)
    return records


def _normalize_memory_snapshot(items: list[Any]) -> list[dict[str, str]]:
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
    return sorted(normalized, key=lambda item: (item["main_class"], item["sub_class"], item["text"]))


def _normalize_message(message: Any) -> dict[str, str]:
    if not isinstance(message, dict):
        raise ValueError("Conversation messages must be objects")
    role = str(message.get("role") or "").strip()
    content = str(message.get("content") or "")
    timestamp = str(message.get("timestamp") or "").strip()
    if not role or not content.strip() or not timestamp:
        raise ValueError("Conversation messages must include role, content, and timestamp")
    return {"role": role, "content": content, "timestamp": timestamp}


def _format_timestamp(base_date: str, offset_index: int) -> str:
    base = datetime.strptime(base_date, _DATE_FORMAT)
    return (base + timedelta(seconds=offset_index * 15)).isoformat(timespec="seconds")


def _materialize_sessions_from_source(
    source_record: dict[str, Any],
    selected_indexes: list[int] | None,
) -> list[dict[str, Any]]:
    haystack_sessions = source_record.get("haystack_sessions")
    haystack_dates = source_record.get("haystack_dates")
    haystack_session_ids = source_record.get("haystack_session_ids")
    if not isinstance(haystack_sessions, list) or not isinstance(haystack_dates, list):
        raise ValueError("LongMemEval source records must include haystack_sessions and haystack_dates")
    if not isinstance(haystack_session_ids, list):
        raise ValueError("LongMemEval source records must include haystack_session_ids")

    indexes = selected_indexes if selected_indexes is not None else list(range(len(haystack_sessions)))
    sessions: list[dict[str, Any]] = []
    for session_index in indexes:
        if not (0 <= session_index < len(haystack_sessions)):
            raise ValueError(f"Invalid session index {session_index} for source question")
        raw_session = haystack_sessions[session_index]
        raw_date = str(haystack_dates[session_index])
        session_id = str(haystack_session_ids[session_index])
        if not isinstance(raw_session, list):
            raise ValueError("LongMemEval source sessions must be message lists")
        conversation = [
            {
                "role": str(message.get("role") or "").strip(),
                "content": str(message.get("content") or ""),
                "timestamp": _format_timestamp(raw_date, offset),
            }
            for offset, message in enumerate(raw_session)
            if isinstance(message, dict)
        ]
        if not conversation:
            continue
        sessions.append({
            "session_id": session_id,
            "session_date": raw_date,
            "conversation": [_normalize_message(message) for message in conversation],
        })
    if not sessions:
        raise ValueError("Materialized case has no sessions")
    return sessions


def _normalize_case(
    case: dict[str, Any],
    source_records: dict[str, dict[str, Any]] | None = None,
    *,
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    case_id = str(case.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("Query eval cases must include case_id")

    level = str(case.get("level") or "").strip().upper()
    if level not in _LEVELS:
        raise ValueError(f"Case {case_id} has unsupported level: {level or '(empty)'}")

    source_question_id = str(case.get("source_question_id") or "").strip()
    source_record = source_records.get(source_question_id) if source_records and source_question_id else None

    source_question_type = str(
        case.get("source_question_type")
        or (source_record or {}).get("question_type")
        or ""
    ).strip()
    if source_question_type not in _QUESTION_TYPES:
        raise ValueError(
            f"Case {case_id} has unsupported source_question_type: {source_question_type or '(empty)'}"
        )

    question = str(case.get("question") or (source_record or {}).get("question") or "").strip()
    question_date = str(case.get("question_date") or (source_record or {}).get("question_date") or "").strip()
    gold_answer = str(case.get("gold_answer") or (source_record or {}).get("answer") or "").strip()
    if not question or not question_date or not gold_answer:
        raise ValueError(
            f"Case {case_id} must include question, question_date, and gold_answer "
            "either directly or through source backfill"
        )

    aliases_raw = case.get("gold_answer_aliases") or []
    if not isinstance(aliases_raw, list):
        raise ValueError(f"Case {case_id} gold_answer_aliases must be a list")

    canonical_snapshot = _normalize_memory_snapshot(case.get("canonical_snapshot") or [])
    if mode == "snapshot" and not canonical_snapshot:
        raise ValueError(f"Case {case_id} must include non-empty canonical_snapshot")

    support_memories = _normalize_memory_snapshot(case.get("gold_support_memories") or [])
    if not support_memories:
        raise ValueError(f"Case {case_id} must include non-empty gold_support_memories")

    replay_sessions: list[dict[str, Any]] = []
    if "replay_sessions" in case:
        raw_sessions = case.get("replay_sessions")
        if not isinstance(raw_sessions, list):
            raise ValueError(f"Case {case_id} replay_sessions must be a list")
        for session in raw_sessions:
            if not isinstance(session, dict):
                raise ValueError(f"Case {case_id} replay_sessions items must be objects")
            session_id = str(session.get("session_id") or "").strip()
            session_date = str(session.get("session_date") or "").strip()
            conversation = session.get("conversation")
            if not session_id or not session_date or not isinstance(conversation, list) or not conversation:
                raise ValueError(f"Case {case_id} has malformed replay session payload")
            replay_sessions.append({
                "session_id": session_id,
                "session_date": session_date,
                "conversation": [_normalize_message(message) for message in conversation],
            })
    elif source_record is not None and "session_indexes" in case:
        session_indexes_raw = case.get("session_indexes")
        if not isinstance(session_indexes_raw, list) or not session_indexes_raw:
            raise ValueError(f"Case {case_id} session_indexes must be a non-empty list")
        replay_sessions = _materialize_sessions_from_source(
            source_record,
            [int(value) for value in session_indexes_raw],
        )

    return {
        "case_id": case_id,
        "level": level,
        "source_question_id": source_question_id,
        "source_question_type": source_question_type,
        "question": question,
        "question_date": question_date,
        "gold_answer": gold_answer,
        "gold_answer_aliases": [str(alias).strip() for alias in aliases_raw if str(alias).strip()],
        "canonical_snapshot": canonical_snapshot,
        "gold_support_memories": support_memories,
        "gold_forbidden_memories": _normalize_memory_snapshot(case.get("gold_forbidden_memories") or []),
        "notes": str(case.get("notes") or "").strip(),
        "source_metadata": dict(case.get("source_metadata") or {}),
        "replay_sessions": replay_sessions,
    }


def load_memory_query_cases(
    cases_path: Path | None = None,
    *,
    mode: str = DEFAULT_MODE,
    level: str = "all",
    question_type: str = "all",
    source_dataset_path: Path | None = None,
) -> list[dict[str, Any]]:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in _MODES:
        raise ValueError(f"Unsupported mode filter: {mode}")

    level_key = level.strip().upper()
    if level_key not in {"ALL", *sorted(_LEVELS)}:
        raise ValueError(f"Unsupported level filter: {level}")

    question_type_key = question_type.strip().lower()
    if question_type_key not in {"all", *_QUESTION_TYPES}:
        raise ValueError(f"Unsupported question_type filter: {question_type}")

    if cases_path is None:
        raise ValueError("Query evaluation datasets are not bundled; supply cases_path explicitly.")
    source = cases_path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Query eval fixtures not found: {source}")

    source_records: dict[str, dict[str, Any]] | None = None
    if source_dataset_path is not None:
        dataset_path = source_dataset_path.resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"LongMemEval source dataset not found: {dataset_path}")
        source_records = _load_source_records(dataset_path)

    raw_cases: list[dict[str, Any]] = []
    if source.is_file():
        raw_cases.extend(_load_case_payload(source))
    else:
        for path in sorted(source.rglob("*.json")):
            raw_cases.extend(_load_case_payload(path))

    normalized_cases = [
        _normalize_case(case, source_records, mode=normalized_mode)
        for case in raw_cases
    ]
    filtered = [
        case
        for case in normalized_cases
        if (level_key == "ALL" or case["level"] == level_key)
        and (question_type_key == "all" or case["source_question_type"] == question_type_key)
    ]
    if not filtered:
        raise ValueError("No query evaluation cases matched the requested filters")
    return filtered


def _normalize_text(value: str) -> str:
    text = value.lower().strip()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[$,]", "", text)
    text = re.sub(r"\b(\d+)\s*:\s*(\d+)\b", r"\1:\2", text)
    text = re.sub(r"\b(\d+)\s+(?:minutes?|mins?)\s+and\s+(\d+)\s+(?:seconds?|secs?)\b", r"\1:\2", text)
    text = re.sub(r"\b(\d+)\s+(?:minutes?|mins?)\b", r"\1 min", text)
    text = re.sub(r"[^\w\s:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_answer_candidates(answer: str, aliases: list[str]) -> set[str]:
    candidates = {_normalize_text(answer)}
    candidates.update(_normalize_text(alias) for alias in aliases if alias.strip())
    return {candidate for candidate in candidates if candidate}


def _deterministic_answer_match(answer: str, gold_answer: str, aliases: list[str]) -> bool:
    candidate = _normalize_text(answer)
    if not candidate:
        return False
    return candidate in _normalize_answer_candidates(gold_answer, aliases)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _bucket_match_memories(
    predicted_items: list[dict[str, str]],
    gold_items: list[dict[str, str]],
    *,
    threshold: float,
    embedder: EmbeddingBackend,
) -> dict[str, Any]:
    if not predicted_items and not gold_items:
        return {"accepted_matches": [], "unmatched_predicted": [], "unmatched_gold": []}
    if not predicted_items:
        return {"accepted_matches": [], "unmatched_predicted": [], "unmatched_gold": list(gold_items)}
    if not gold_items:
        return {"accepted_matches": [], "unmatched_predicted": list(predicted_items), "unmatched_gold": []}

    matched_predicted: set[int] = set()
    matched_gold: set[int] = set()
    accepted_matches: list[dict[str, Any]] = []

    gold_by_norm: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(gold_items):
        gold_by_norm[_normalize_text(item["text"])].append(index)

    for predicted_index, predicted in enumerate(predicted_items):
        candidate_indexes = gold_by_norm.get(_normalize_text(predicted["text"]), [])
        while candidate_indexes and candidate_indexes[0] in matched_gold:
            candidate_indexes.pop(0)
        if not candidate_indexes:
            continue
        gold_index = candidate_indexes.pop(0)
        matched_predicted.add(predicted_index)
        matched_gold.add(gold_index)
        accepted_matches.append({
            "main_class": predicted["main_class"],
            "predicted": predicted,
            "gold": gold_items[gold_index],
            "similarity": 1.0,
            "match_stage": "exact",
        })

    remaining_predicted = [
        item for index, item in enumerate(predicted_items) if index not in matched_predicted
    ]
    remaining_gold = [item for index, item in enumerate(gold_items) if index not in matched_gold]
    if not remaining_predicted or not remaining_gold:
        return {
            "accepted_matches": sorted(
                accepted_matches,
                key=lambda item: (
                    item["main_class"],
                    item["gold"]["sub_class"],
                    item["gold"]["text"],
                    item["predicted"]["sub_class"],
                    item["predicted"]["text"],
                ),
            ),
            "unmatched_predicted": remaining_predicted,
            "unmatched_gold": remaining_gold,
        }

    predicted_vectors = embedder.encode_texts([item["text"] for item in remaining_predicted])
    gold_vectors = embedder.encode_texts([item["text"] for item in remaining_gold])
    similarity_matrix = [
        [_cosine_similarity(predicted_vector, gold_vector) for gold_vector in gold_vectors]
        for predicted_vector in predicted_vectors
    ]

    linear_sum_assignment = _load_hungarian()
    row_indexes, col_indexes = linear_sum_assignment(
        [[1.0 - score for score in row] for row in similarity_matrix]
    )

    semantic_matched_predicted: set[int] = set()
    semantic_matched_gold: set[int] = set()
    for row_index, col_index in zip(list(row_indexes), list(col_indexes)):
        similarity = float(similarity_matrix[int(row_index)][int(col_index)])
        if similarity < threshold:
            continue
        semantic_matched_predicted.add(int(row_index))
        semantic_matched_gold.add(int(col_index))
        accepted_matches.append({
            "main_class": remaining_predicted[int(row_index)]["main_class"],
            "predicted": remaining_predicted[int(row_index)],
            "gold": remaining_gold[int(col_index)],
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
            item for index, item in enumerate(remaining_predicted) if index not in semantic_matched_predicted
        ],
        "unmatched_gold": [
            item for index, item in enumerate(remaining_gold) if index not in semantic_matched_gold
        ],
    }


def _match_memory_snapshots(
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
        bucket = _bucket_match_memories(
            sorted(predicted_by_class.get(main_class, []), key=lambda item: (item["sub_class"], item["text"])),
            sorted(gold_by_class.get(main_class, []), key=lambda item: (item["sub_class"], item["text"])),
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


async def _judge_answer(
    *,
    provider: Any,
    model: str,
    question: str,
    gold_answer: str,
    candidate_answer: str,
    source_question_type: str,
) -> bool:
    messages = [
        {
            "role": "system",
            "content": (
                "You are grading whether a candidate answer is correct for a memory retrieval benchmark. "
                "Return JSON only: {\"verdict\":\"correct\"} or {\"verdict\":\"incorrect\"}. "
                "Treat semantically equivalent answers as correct. For knowledge-update cases, only the latest "
                "value counts. For single-session-preference cases, preserve the actual preference constraint."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "candidate_answer": candidate_answer,
                    "source_question_type": source_question_type,
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = await provider.chat_with_retry(messages=messages, model=model, temperature=0.0)
    content = (response.content or "").strip()
    if not content:
        return False
    try:
        payload = json.loads(content)
        return str(payload.get("verdict") or "").strip().lower() == "correct"
    except Exception:
        lowered = content.lower()
        if "incorrect" in lowered:
            return False
        return "correct" in lowered


def _score_answer(
    *,
    gold_answer: str,
    aliases: list[str],
    candidate_answer: str,
    judge_result: bool | None,
) -> dict[str, Any]:
    if _deterministic_answer_match(candidate_answer, gold_answer, aliases):
        return {"answer_exact": True, "answer_correct": True, "answer_score_method": "exact"}
    return {
        "answer_exact": False,
        "answer_correct": bool(judge_result),
        "answer_score_method": "judge",
    }


def _core_memories(snapshot: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in snapshot if item["main_class"] in _CORE_CLASSES]


def _seed_snapshot(workspace: Path, snapshot: list[dict[str, str]]) -> list[dict[str, str]]:
    db = MemoryDatabase(workspace)
    db.initialize()
    normalized_snapshot = _normalize_memory_snapshot(snapshot)
    records = tuple(
        MemoryRecord.create(item["main_class"], item["sub_class"], item["text"])
        for item in normalized_snapshot
    )
    db.commit_snapshot(
        MemorySnapshot(0, records), expected_revision=0, event_id="eval-seed",
        ts="1970-01-01T00:00:00", session_key="eval", history_text="",
        candidate_type="fixture",
    )
    MemorySynchronizer(db).sync()
    return normalized_snapshot


async def _evaluate_seeded_workspace(
    *,
    workspace: Path,
    case: dict[str, Any],
    answer_provider: Any,
    judge_provider: Any,
    model: str,
    judge_model: str,
    top_k: int,
    threshold: float,
    embedder: EmbeddingBackend,
    retrieval_source_mode: str,
    replay_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    db = MemoryDatabase(workspace)
    final_snapshot = _normalize_memory_snapshot(db.list_canonical_memories())
    retrieved_memories = _normalize_memory_snapshot(
        db.query_canonical_memories(case["question"], limit=top_k)
    )
    core_memories = _core_memories(final_snapshot)

    service = MemoryService(workspace, answer_provider, model, database=db)
    builder = ContextBuilder(workspace)
    memory_context = await service.prepare_context(case["question"], retrieval_budget=top_k)
    answer_messages = builder.build_messages(
        history=[], current_message=case["question"], memory_context=memory_context,
    )
    answer_response = await answer_provider.chat_with_retry(messages=answer_messages, model=model)
    generated_answer = (answer_response.content or "").strip()

    retrieval_matching = _match_memory_snapshots(
        retrieved_memories,
        case["gold_support_memories"],
        threshold=threshold,
        embedder=embedder,
    )
    core_matching = _match_memory_snapshots(
        core_memories,
        case["gold_support_memories"],
        threshold=threshold,
        embedder=embedder,
    )
    stale_matching = _match_memory_snapshots(
        retrieved_memories,
        case["gold_forbidden_memories"],
        threshold=threshold,
        embedder=embedder,
    )

    judge_result: bool | None = None
    if not _deterministic_answer_match(
        generated_answer,
        case["gold_answer"],
        case["gold_answer_aliases"],
    ):
        judge_result = await _judge_answer(
            provider=judge_provider,
            model=judge_model,
            question=case["question"],
            gold_answer=case["gold_answer"],
            candidate_answer=generated_answer,
            source_question_type=case["source_question_type"],
        )

    answer_metrics = _score_answer(
        gold_answer=case["gold_answer"],
        aliases=case["gold_answer_aliases"],
        candidate_answer=generated_answer,
        judge_result=judge_result,
    )

    matched_retrieved_indexes: set[int] = set()
    first_hit_rank: int | None = None
    for match in retrieval_matching["accepted_matches"]:
        predicted = match["predicted"]
        for index, retrieved in enumerate(retrieved_memories):
            if retrieved == predicted:
                matched_retrieved_indexes.add(index)
                if first_hit_rank is None or index < first_hit_rank:
                    first_hit_rank = index
                break

    all_support_retrieved = not retrieval_matching["unmatched_gold"]
    all_support_in_core = not core_matching["unmatched_gold"]
    stale_exposed = bool(stale_matching["accepted_matches"])
    support_hit_at_1 = 0 in matched_retrieved_indexes
    support_hit_at_k = bool(matched_retrieved_indexes)
    support_recall_at_k = (
        len(retrieval_matching["accepted_matches"]) / len(case["gold_support_memories"])
        if case["gold_support_memories"]
        else 1.0
    )
    support_mrr = (1.0 / (first_hit_rank + 1)) if first_hit_rank is not None else 0.0
    retrieval_exact_match = all_support_retrieved and not stale_exposed
    retrieval_bucket = (
        ("hit" if all_support_retrieved else "miss")
        + "_"
        + ("correct" if answer_metrics["answer_correct"] else "wrong")
    )

    result = {
        "case_id": case["case_id"],
        "level": case["level"],
        "source_question_id": case["source_question_id"],
        "source_question_type": case["source_question_type"],
        "question": case["question"],
        "question_date": case["question_date"],
        "gold_answer": case["gold_answer"],
        "gold_answer_aliases": case["gold_answer_aliases"],
        "canonical_snapshot": case["canonical_snapshot"],
        "generated_answer": generated_answer,
        "gold_support_memories": case["gold_support_memories"],
        "gold_forbidden_memories": case["gold_forbidden_memories"],
        "final_snapshot": final_snapshot,
        "core_memories": core_memories,
        "retrieved_memories": retrieved_memories,
        "retrieval_matches": retrieval_matching["accepted_matches"],
        "missing_support_memories": retrieval_matching["unmatched_gold"],
        "stale_retrieved_memories": [match["predicted"] for match in stale_matching["accepted_matches"]],
        "support_available_in_core": all_support_in_core,
        "support_available_in_retrieved": all_support_retrieved,
        "retrieval_source_mode": retrieval_source_mode,
        "metrics": {
            "support_hit_at_1": support_hit_at_1,
            "support_hit_at_k": support_hit_at_k,
            "support_recall_at_k": support_recall_at_k,
            "support_mrr": support_mrr,
            "stale_exposed": stale_exposed,
            "retrieval_exact_match": retrieval_exact_match,
            "answer_accuracy": answer_metrics["answer_correct"],
            "answer_exact": answer_metrics["answer_exact"],
            "answer_score_method": answer_metrics["answer_score_method"],
            "retrieval_bucket": retrieval_bucket,
        },
    }
    if replay_metrics is not None:
        result["replay_diagnostics"] = replay_metrics
    return result


async def _run_snapshot_query_case(
    *,
    case: dict[str, Any],
    answer_provider: Any,
    judge_provider: Any,
    model: str,
    judge_model: str,
    top_k: int,
    threshold: float,
    embedder: EmbeddingBackend,
) -> dict[str, Any]:
    with TemporaryDirectory(prefix=f"nanobot-memory-query-{case['case_id']}-") as tmp:
        workspace = Path(tmp)
        _seed_snapshot(workspace, case["canonical_snapshot"])
        return await _evaluate_seeded_workspace(
            workspace=workspace,
            case=case,
            answer_provider=answer_provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            top_k=top_k,
            threshold=threshold,
            embedder=embedder,
            retrieval_source_mode="snapshot",
        )


async def _run_replay_query_case(
    *,
    case: dict[str, Any],
    consolidate_provider: Any,
    answer_provider: Any,
    judge_provider: Any,
    model: str,
    judge_model: str,
    top_k: int,
    threshold: float,
    embedder: EmbeddingBackend,
) -> dict[str, Any]:
    if not case["replay_sessions"]:
        raise ValueError(f"Replay mode requires replay_sessions for case {case['case_id']}")

    with TemporaryDirectory(prefix=f"nanobot-memory-query-{case['case_id']}-") as tmp:
        workspace = Path(tmp)
        service = MemoryService(workspace, consolidate_provider, model)
        consolidate_results: list[bool] = []
        for session in case["replay_sessions"]:
            result = await service.consolidate(session["conversation"], session_key="eval")
            consolidate_results.append(result.database_committed)
        return await _evaluate_seeded_workspace(
            workspace=workspace,
            case=case,
            answer_provider=answer_provider,
            judge_provider=judge_provider,
            model=model,
            judge_model=judge_model,
            top_k=top_k,
            threshold=threshold,
            embedder=embedder,
            retrieval_source_mode="replay",
            replay_metrics={
                "session_count": len(case["replay_sessions"]),
                "consolidate_success_rate": (
                    sum(int(value is True) for value in consolidate_results) / len(consolidate_results)
                    if consolidate_results
                    else 1.0
                ),
            },
        )


def _aggregate_group(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(case_results)
    if not count:
        return {
            "case_count": 0,
            "support_hit_at_1": 0.0,
            "support_hit_at_k": 0.0,
            "support_recall_at_k": 0.0,
            "support_mrr": 0.0,
            "stale_exposed_rate": None,
            "retrieval_exact_match_rate": 0.0,
            "answer_accuracy": 0.0,
            "answer_exact_rate": 0.0,
            "answer_judge_pass_rate": 0.0,
            "unsupported_but_correct_rate": 0.0,
            "core_only_rescue_rate": 0.0,
            "retrieval_answer_consistency": {
                "hit_correct": 0,
                "hit_wrong": 0,
                "miss_correct": 0,
                "miss_wrong": 0,
            },
        }

    stale_candidates = [
        result
        for result in case_results
        if result["source_question_type"] == "knowledge-update"
        and result["gold_forbidden_memories"]
    ]
    consistency = {
        "hit_correct": 0,
        "hit_wrong": 0,
        "miss_correct": 0,
        "miss_wrong": 0,
    }
    for result in case_results:
        consistency[result["metrics"]["retrieval_bucket"]] += 1

    judge_passes = sum(
        int(
            result["metrics"]["answer_score_method"] == "judge"
            and result["metrics"]["answer_accuracy"]
        )
        for result in case_results
    )
    unsupported_but_correct = sum(
        int((not result["support_available_in_retrieved"]) and result["metrics"]["answer_accuracy"])
        for result in case_results
    )
    core_only_rescue = sum(
        int(
            (not result["support_available_in_retrieved"])
            and result["support_available_in_core"]
            and result["metrics"]["answer_accuracy"]
        )
        for result in case_results
    )
    return {
        "case_count": count,
        "support_hit_at_1": sum(int(result["metrics"]["support_hit_at_1"]) for result in case_results) / count,
        "support_hit_at_k": sum(int(result["metrics"]["support_hit_at_k"]) for result in case_results) / count,
        "support_recall_at_k": sum(result["metrics"]["support_recall_at_k"] for result in case_results) / count,
        "support_mrr": sum(result["metrics"]["support_mrr"] for result in case_results) / count,
        "stale_exposed_rate": (
            sum(int(result["metrics"]["stale_exposed"]) for result in stale_candidates) / len(stale_candidates)
            if stale_candidates
            else None
        ),
        "retrieval_exact_match_rate": (
            sum(int(result["metrics"]["retrieval_exact_match"]) for result in case_results) / count
        ),
        "answer_accuracy": sum(int(result["metrics"]["answer_accuracy"]) for result in case_results) / count,
        "answer_exact_rate": sum(int(result["metrics"]["answer_exact"]) for result in case_results) / count,
        "answer_judge_pass_rate": judge_passes / count,
        "unsupported_but_correct_rate": unsupported_but_correct / count,
        "core_only_rescue_rate": core_only_rescue / count,
        "retrieval_answer_consistency": consistency,
    }


async def _run_memory_v2_query_eval_async(
    cases: list[dict[str, Any]],
    *,
    mode: str,
    provider: Any | None,
    provider_factory: Any | None,
    answer_provider: Any | None,
    answer_provider_factory: Any | None,
    judge_provider: Any | None,
    judge_provider_factory: Any | None,
    model: str,
    judge_model: str,
    cases_source: str,
    source_dataset: str | None,
    top_k: int,
    embedding_model: str,
    threshold: float,
    embedder: EmbeddingBackend,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("At least one query evaluation case is required")
    if mode not in _MODES:
        raise ValueError(f"Unsupported query evaluation mode: {mode}")

    if provider is not None and provider_factory is not None:
        raise ValueError("Provide either provider or provider_factory, not both")
    if answer_provider is not None and answer_provider_factory is not None:
        raise ValueError("Provide either answer_provider or answer_provider_factory, not both")
    if judge_provider is not None and judge_provider_factory is not None:
        raise ValueError("Provide either judge_provider or judge_provider_factory, not both")

    if mode == "replay" and provider is None and provider_factory is None:
        raise ValueError("Replay mode requires provider or provider_factory")
    if mode == "snapshot" and provider is not None and answer_provider is None and answer_provider_factory is None:
        # allowed because CLI may pass provider as answer_provider explicitly, but disallow silent fallback here
        pass
    if answer_provider is None and answer_provider_factory is None:
        raise ValueError("Query evaluation requires answer_provider or answer_provider_factory")

    case_results: list[dict[str, Any]] = []
    for case in cases:
        resolved_answer_provider = (
            answer_provider_factory(case)
            if answer_provider_factory is not None
            else answer_provider
        )
        if resolved_answer_provider is None:
            raise ValueError("Resolved answer provider is missing")

        resolved_judge_provider = (
            judge_provider_factory(case)
            if judge_provider_factory is not None
            else (judge_provider or resolved_answer_provider)
        )

        if mode == "snapshot":
            case_results.append(
                await _run_snapshot_query_case(
                    case=case,
                    answer_provider=resolved_answer_provider,
                    judge_provider=resolved_judge_provider,
                    model=model,
                    judge_model=judge_model,
                    top_k=top_k,
                    threshold=threshold,
                    embedder=embedder,
                )
            )
        else:
            consolidate_provider = provider_factory(case) if provider_factory is not None else provider
            case_results.append(
                await _run_replay_query_case(
                    case=case,
                    consolidate_provider=consolidate_provider,
                    answer_provider=resolved_answer_provider,
                    judge_provider=resolved_judge_provider,
                    model=model,
                    judge_model=judge_model,
                    top_k=top_k,
                    threshold=threshold,
                    embedder=embedder,
                )
            )

    by_level = {
        level: _aggregate_group([result for result in case_results if result["level"] == level])
        for level in sorted(_LEVELS)
        if any(result["level"] == level for result in case_results)
    }
    by_question_type = {
        qtype: _aggregate_group(
            [result for result in case_results if result["source_question_type"] == qtype]
        )
        for qtype in sorted(_QUESTION_TYPES)
        if any(result["source_question_type"] == qtype for result in case_results)
    }

    return {
        "report_version": REPORT_VERSION,
        "evaluation": "memory_v2_query",
        "cases_source": cases_source,
        "source_dataset": source_dataset,
        "mode": mode,
        "model": model,
        "judge_model": judge_model,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "threshold": threshold,
        "total_cases": len(case_results),
        "summary": {
            "overall": _aggregate_group(case_results),
            "by_level": by_level,
            "by_question_type": by_question_type,
        },
        "cases": case_results,
    }


def run_memory_v2_query_eval(
    *,
    mode: str = DEFAULT_MODE,
    provider: Any | None = None,
    provider_factory: Any | None = None,
    answer_provider: Any | None = None,
    answer_provider_factory: Any | None = None,
    judge_provider: Any | None = None,
    judge_provider_factory: Any | None = None,
    model: str,
    judge_model: str | None = None,
    cases: list[dict[str, Any]] | None = None,
    cases_path: Path | None = None,
    level: str = "all",
    question_type: str = "all",
    source_dataset_path: Path | None = None,
    top_k: int = DEFAULT_TOP_K,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    threshold: float = DEFAULT_RETRIEVAL_THRESHOLD,
    embedder: EmbeddingBackend | None = None,
) -> dict[str, Any]:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in _MODES:
        raise ValueError(f"Unsupported query evaluation mode: {mode}")
    resolved_judge_model = judge_model.strip() if judge_model and judge_model.strip() else model

    source_dataset = source_dataset_path.resolve() if source_dataset_path else None
    source_records = _load_source_records(source_dataset) if source_dataset is not None else None
    if cases is not None:
        loaded_cases = [
            _normalize_case(case, source_records, mode=normalized_mode)
            for case in cases
        ]
    else:
        loaded_cases = load_memory_query_cases(
            cases_path,
            mode=normalized_mode,
            level=level,
            question_type=question_type,
            source_dataset_path=source_dataset,
        )
    cases_source = (
        str(cases_path.resolve())
        if cases_path
        else "inline"
    )
    resolved_embedder = embedder or _SentenceTransformerBackend(embedding_model)
    return asyncio.run(
        _run_memory_v2_query_eval_async(
            loaded_cases,
            mode=normalized_mode,
            provider=provider,
            provider_factory=provider_factory,
            answer_provider=answer_provider,
            answer_provider_factory=answer_provider_factory,
            judge_provider=judge_provider,
            judge_provider_factory=judge_provider_factory,
            model=model,
            judge_model=resolved_judge_model,
            cases_source=cases_source,
            source_dataset=str(source_dataset) if source_dataset is not None else None,
            top_k=top_k,
            embedding_model=embedding_model,
            threshold=threshold,
            embedder=resolved_embedder,
        )
    )


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_memory_item(item: dict[str, Any]) -> str:
    return f"[{item['main_class']}/{item['sub_class']}] {item['text']}"


def render_memory_v2_query_report_markdown(report: dict[str, Any]) -> str:
    overall = report["summary"]["overall"]
    consistency = overall["retrieval_answer_consistency"]
    lines = [
        "# V2 Memory Query Evaluation Report",
        "",
        f"- Report Version: {report['report_version']}",
        f"- Cases Source: `{report['cases_source']}`",
        f"- Source Dataset: `{report['source_dataset'] or '(not used)'}`",
        f"- Total Cases: {report['total_cases']}",
        f"- Retrieval Source Mode: `{report['mode']}`",
        f"- Model: `{report['model']}`",
        f"- Judge Model: `{report['judge_model']}`",
        f"- Embedding Model: `{report['embedding_model']}`",
        f"- Top K: `{report['top_k']}`",
        f"- Retrieval Threshold: `{report['threshold']:.2f}`",
        "",
        "## Overall",
        "",
        "| Hit@1 | Hit@K | Recall@K | MRR | Exact Match | Stale Exposed | Answer Acc | Answer Exact | Judge Pass | Core Rescue | Unsupported but Correct |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {_format_metric(overall['support_hit_at_1'])} | {_format_metric(overall['support_hit_at_k'])} | "
            f"{_format_metric(overall['support_recall_at_k'])} | {_format_metric(overall['support_mrr'])} | "
            f"{_format_metric(overall['retrieval_exact_match_rate'])} | {_format_metric(overall['stale_exposed_rate'])} | "
            f"{_format_metric(overall['answer_accuracy'])} | {_format_metric(overall['answer_exact_rate'])} | "
            f"{_format_metric(overall['answer_judge_pass_rate'])} | {_format_metric(overall['core_only_rescue_rate'])} | "
            f"{_format_metric(overall['unsupported_but_correct_rate'])} |"
        ),
        "",
        "## Retrieval vs Answer",
        "",
        f"- retrieve_hit + answer_correct: `{consistency['hit_correct']}`",
        f"- retrieve_hit + answer_wrong: `{consistency['hit_wrong']}`",
        f"- retrieve_miss + answer_correct: `{consistency['miss_correct']}`",
        f"- retrieve_miss + answer_wrong: `{consistency['miss_wrong']}`",
        "",
        "## Breakdown",
        "",
        "| Group | Cases | Hit@K | Recall@K | MRR | Stale Exposed | Answer Acc | Core Rescue |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for prefix, groups in (
        ("level", report["summary"]["by_level"]),
        ("question_type", report["summary"]["by_question_type"]),
    ):
        for name, metrics in groups.items():
            lines.append(
                f"| {prefix}:{name} | {metrics['case_count']} | {_format_metric(metrics['support_hit_at_k'])} | "
                f"{_format_metric(metrics['support_recall_at_k'])} | {_format_metric(metrics['support_mrr'])} | "
                f"{_format_metric(metrics['stale_exposed_rate'])} | {_format_metric(metrics['answer_accuracy'])} | "
                f"{_format_metric(metrics['core_only_rescue_rate'])} |"
            )

    lines.extend([
        "",
        "## Case Results",
        "",
        "| Case | Level | Question Type | Source Mode | Retrieve Support | Core Support | Stale | Answer | Method |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for case in report["cases"]:
        metrics = case["metrics"]
        lines.append(
            f"| {case['case_id']} | {case['level']} | {case['source_question_type']} | "
            f"{case['retrieval_source_mode']} | {case['support_available_in_retrieved']} | "
            f"{case['support_available_in_core']} | {metrics['stale_exposed']} | "
            f"{metrics['answer_accuracy']} | {metrics['answer_score_method']} |"
        )

    lines.extend(["", "## Failed Case Details"])
    failed_cases = [
        case
        for case in report["cases"]
        if (not case["support_available_in_retrieved"]) or (not case["metrics"]["answer_accuracy"])
    ]
    if not failed_cases:
        lines.extend(["", "- (none)"])
        return "\n".join(lines)

    for case in failed_cases:
        lines.extend([
            "",
            f"### {case['case_id']}",
            "",
            f"- Source Mode: `{case['retrieval_source_mode']}`",
            f"- Question Type: `{case['source_question_type']}`",
            f"- Question: {case['question']}",
            f"- Gold Answer: `{case['gold_answer']}`",
            f"- Generated Answer: `{case['generated_answer']}`",
            f"- Retrieved Support Available: `{case['support_available_in_retrieved']}`",
            f"- Core Support Available: `{case['support_available_in_core']}`",
            f"- Answer Correct: `{case['metrics']['answer_accuracy']}`",
            "- Canonical Snapshot:",
        ])
        if case["canonical_snapshot"]:
            lines.extend([f"  - {_format_memory_item(item)}" for item in case["canonical_snapshot"]])
        else:
            lines.append("  - (none)")
        lines.append("- Missing Support Memories:")
        if case["missing_support_memories"]:
            lines.extend([f"  - {_format_memory_item(item)}" for item in case["missing_support_memories"]])
        else:
            lines.append("  - (none)")
        lines.append("- Stale Retrieved Memories:")
        if case["stale_retrieved_memories"]:
            lines.extend([f"  - {_format_memory_item(item)}" for item in case["stale_retrieved_memories"]])
        else:
            lines.append("  - (none)")
        if case.get("replay_diagnostics"):
            lines.append("- Replay Diagnostics:")
            lines.extend([f"  - {key}: `{value}`" for key, value in case["replay_diagnostics"].items()])

    return "\n".join(lines)


def save_memory_v2_query_report(report: dict[str, Any], save_dir: Path) -> dict[str, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "memory_v2_query_eval_report.json"
    markdown_path = save_dir / "memory_v2_query_eval_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_memory_v2_query_report_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
