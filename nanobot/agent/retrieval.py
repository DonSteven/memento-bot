"""Shared retrieval primitives and DashScope model backends."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from typing import Any, Hashable, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx

FTS_TOKENIZER_VERSION = "unicode-bigram-v1"
_RRF_K = 60
_ID = TypeVar("_ID", bound=Hashable)


class EmbeddingBackend(Protocol):
    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class RerankerBackend(Protocol):
    async def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def load_sqlite_vec(conn: sqlite3.Connection, *, feature: str) -> None:
    try:
        import sqlite_vec
    except ImportError as exc:
        raise RuntimeError(
            f"{feature} requires sqlite-vec. Install it with `uv sync --extra web_knowledge`."
        ) from exc
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.Error) as exc:
        raise RuntimeError(
            "The active sqlite3 build cannot enable extension loading required by sqlite-vec."
        ) from exc
    try:
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.Error) as exc:
        raise RuntimeError(f"Failed to load sqlite-vec extension: {exc}") from exc
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass


def fts_tokens(text: str) -> tuple[str, ...]:
    """Tokenize English words and CJK runs for identical index/query behavior."""
    tokens: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+", text):
        raw = match.group(0).lower()
        candidates: list[str]
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", raw):
            candidates = (
                list(raw) if len(raw) == 1 else [raw[i : i + 2] for i in range(len(raw) - 1)]
            )
            # Single characters make one-character queries reachable without weakening query rules.
            candidates.extend(raw)
        else:
            candidates = [raw.strip("_")]
            for suffix in ("ly", "ing", "ed", "es", "s"):
                if raw.endswith(suffix) and len(raw) - len(suffix) >= 2:
                    candidates.append(raw[: -len(suffix)].strip("_"))
        for token in candidates:
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
    return tuple(tokens)


def fts_index_text(text: str) -> str:
    return " ".join(fts_tokens(text))


def build_fts_query(query: str) -> str:
    return " OR ".join(f'"{token}"' for token in lexical_query_terms(query))


def lexical_query_terms(query: str) -> tuple[str, ...]:
    """Keep unigrams for single-character Chinese queries only."""
    terms = fts_tokens(query)
    if len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", query)) > 1:
        return tuple(term for term in terms if not re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]", term))
    return terms


def passes_lexical_gate(query: str, document: str) -> bool:
    """Require at least half of the distinct meaningful query terms to match."""
    terms = set(lexical_query_terms(query))
    return bool(terms) and len(terms.intersection(fts_tokens(document))) / len(terms) >= 0.5


def reciprocal_rank_fuse(
    result_sets: dict[str, list[dict[str, Any]]], *, key_field: str, limit: int
) -> list[dict[str, Any]]:
    fused: dict[Hashable, dict[str, Any]] = {}
    for source_name, rows in result_sets.items():
        for rank, row in enumerate(rows, start=1):
            key = row[key_field]
            current = fused.setdefault(key, {**row, "sources": [], "rrf_score": 0.0})
            current["rrf_score"] = float(current["rrf_score"]) + 1.0 / (_RRF_K + rank)
            current[f"{source_name}_rank"] = rank
            if source_name not in current["sources"]:
                current["sources"].append(source_name)
            for item_key, item_value in row.items():
                current.setdefault(item_key, item_value)

    def stable_id(value: Any) -> tuple[int, Any]:
        return (0, value) if isinstance(value, int) else (1, str(value))

    return sorted(
        fused.values(),
        key=lambda item: (-float(item["rrf_score"]), stable_id(item[key_field])),
    )[:limit]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def vector_json(vector: list[float]) -> str:
    return json.dumps([float(value) for value in vector], separators=(",", ":"))


def _dashscope_endpoint(api_base: str, path: str) -> str:
    parsed = urlsplit(api_base.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid DashScope API base URL: {api_base!r}")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _response_json(response: httpx.Response, operation: str) -> dict[str, Any]:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text.strip()[:500]
        raise RuntimeError(
            f"DashScope {operation} request failed with HTTP {response.status_code}: {detail}"
        ) from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"DashScope {operation} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"DashScope {operation} returned an invalid response object")
    return payload


def _normalized_vector(raw: Any, dimension: int) -> list[float]:
    if not isinstance(raw, list) or len(raw) != dimension:
        actual = len(raw) if isinstance(raw, list) else "invalid"
        raise RuntimeError(
            f"DashScope embedding dimension mismatch: expected {dimension}, got {actual}"
        )
    vector = [float(value) for value in raw]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise RuntimeError("DashScope embedding returned a zero vector")
    return [value / norm for value in vector]


class DashScopeEmbeddingBackend:
    _MAX_BATCH_SIZE = 10

    def __init__(self, *, api_key: str, api_base: str, model: str, dimension: int):
        if not api_key:
            raise ValueError("Embedding requires providers.dashscope.apiKey")
        self.api_key = api_key
        self.model = model
        self._dimension = dimension
        self.endpoint = _dashscope_endpoint(
            api_base, "/api/v1/services/embeddings/text-embedding/text-embedding"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, text_type="document")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], text_type="query"))[0]

    async def _embed(self, texts: list[str], *, text_type: str) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            for start in range(0, len(texts), self._MAX_BATCH_SIZE):
                batch = texts[start : start + self._MAX_BATCH_SIZE]
                try:
                    response = await client.post(
                        self.endpoint,
                        headers=headers,
                        json={
                            "model": self.model,
                            "input": {"texts": batch},
                            "parameters": {
                                "dimension": self.dimension,
                                "output_type": "dense",
                                "text_type": text_type,
                            },
                        },
                    )
                except httpx.RequestError as exc:
                    raise RuntimeError(f"DashScope embedding request failed: {exc}") from exc
                payload = _response_json(response, "embedding")
                output = payload.get("output")
                items = output.get("embeddings") if isinstance(output, dict) else None
                if not isinstance(items, list) or len(items) != len(batch):
                    raise RuntimeError(
                        "DashScope embedding returned an unexpected number of vectors"
                    )
                by_index: dict[int, list[float]] = {}
                for item in items:
                    if not isinstance(item, dict):
                        raise RuntimeError("DashScope embedding returned an invalid vector item")
                    index = item.get("text_index")
                    if not isinstance(index, int) or index in by_index:
                        raise RuntimeError("DashScope embedding returned an invalid text_index")
                    by_index[index] = _normalized_vector(item.get("embedding"), self.dimension)
                if set(by_index) != set(range(len(batch))):
                    raise RuntimeError(
                        "DashScope embedding response indexes do not match the input"
                    )
                vectors.extend(by_index[index] for index in range(len(batch)))
        return vectors


class DashScopeRerankerBackend:
    def __init__(self, *, api_key: str, api_base: str, model: str, instruct: str):
        if not api_key:
            raise ValueError("Rerank requires providers.dashscope.apiKey")
        self.api_key = api_key
        self.model = model
        self.instruct = instruct
        self.endpoint = _dashscope_endpoint(api_base, "/compatible-api/v1/reranks")

    async def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        if any(pair_query != query for pair_query, _ in pairs):
            raise ValueError("DashScope rerank requires all pairs to share one query")
        body: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [d for _, d in pairs],
            "top_n": len(pairs),
        }
        if self.instruct:
            body["instruct"] = self.instruct
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
        except httpx.RequestError as exc:
            raise RuntimeError(f"DashScope rerank request failed: {exc}") from exc
        payload = _response_json(response, "rerank")
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(pairs):
            raise RuntimeError("DashScope rerank returned an unexpected number of scores")
        scores: dict[int, float] = {}
        for item in results:
            if not isinstance(item, dict):
                raise RuntimeError("DashScope rerank returned an invalid result item")
            index = item.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(pairs) or index in scores:
                raise RuntimeError("DashScope rerank returned an invalid document index")
            scores[index] = float(item.get("relevance_score"))
        if set(scores) != set(range(len(pairs))):
            raise RuntimeError("DashScope rerank response indexes do not match the input")
        return [scores[index] for index in range(len(pairs))]
