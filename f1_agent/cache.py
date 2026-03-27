"""Semantic answer cache backed by FAISS ANN + SQLite.

Embeds user questions with the same Gemini embedding model used for RAG,
stores entries in SQLite, and keeps an in-memory approximate nearest-neighbor
index for sublinear lookup as cache size grows.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import faiss
import numpy as np
from decouple import config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from f1_agent.resilience import run_with_retry

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "f1_cache"

_EMBEDDING_MODEL = cast(
    str,
    config(
        "GEMINI_EMBEDDING_MODEL",
        default="models/gemini-embedding-2-preview",
        cast=str,
    ),
)

# Similarity threshold — 1.0 = identical, lower = more lenient
_SIMILARITY_THRESHOLD = cast(
    float,
    config("F1_SEMANTIC_CACHE_SIMILARITY_THRESHOLD", default=0.92, cast=float),
)

_TOP_K = cast(int, config("F1_SEMANTIC_CACHE_TOP_K", default=8, cast=int))
_HNSW_M = cast(int, config("F1_SEMANTIC_CACHE_HNSW_M", default=32, cast=int))
_HNSW_EF_SEARCH = cast(
    int,
    config("F1_SEMANTIC_CACHE_HNSW_EF_SEARCH", default=64, cast=int),
)
_SWEEP_INTERVAL_S = cast(
    int,
    config("F1_SEMANTIC_CACHE_SWEEP_INTERVAL_S", default=600, cast=int),
)
_SWEEP_EVERY_OPS = cast(
    int,
    config("F1_SEMANTIC_CACHE_SWEEP_EVERY_OPS", default=500, cast=int),
)
_MAX_ENTRIES = cast(
    int,
    config("F1_SEMANTIC_CACHE_MAX_ENTRIES", default=50000, cast=int),
)

# TTLs in seconds
_TTL_STATIC = 30 * 24 * 3600  # 30 days  (historical DB + regulations)
_TTL_WEB = 24 * 3600  # 24 hours (google search results)


@dataclass
class CacheLookupResult:
    answer: str | None
    outcome: str
    lookup_ms: float
    candidates_scanned: int
    similarity_top1: float | None
    evicted_count: int


class _EmbeddingsLike(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key: str = cast(str, config("GEMINI_API_KEY", default="", cast=str))
    if not api_key:
        api_key = cast(str, config("GOOGLE_API_KEY", default="", cast=str))
    if not api_key:
        raise ValueError("API key required for cache embeddings.")

    return GoogleGenerativeAIEmbeddings(
        model=_EMBEDDING_MODEL,
        api_key=SecretStr(api_key),
    )


class SemanticCache:
    """Question-level semantic cache with FAISS ANN similarity search."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        embeddings: _EmbeddingsLike | None = None,
        *,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        top_k: int = _TOP_K,
        hnsw_m: int = _HNSW_M,
        hnsw_ef_search: int = _HNSW_EF_SEARCH,
        sweep_interval_seconds: int = _SWEEP_INTERVAL_S,
        sweep_every_ops: int = _SWEEP_EVERY_OPS,
        max_entries: int = _MAX_ENTRIES,
    ):
        self._dir = cache_dir or _CACHE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

        self._similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self._top_k = max(1, top_k)
        self._hnsw_m = max(4, hnsw_m)
        self._hnsw_ef_search = max(8, hnsw_ef_search)
        self._sweep_interval_seconds = max(1, sweep_interval_seconds)
        self._sweep_every_ops = max(1, sweep_every_ops)
        self._max_entries = max(1, max_entries)

        self._db_path = self._dir / "cache.db"
        self._embeddings = embeddings if embeddings is not None else _get_embeddings()
        self._lock = threading.RLock()
        self._conn = self._init_db()
        self._vector_dim: int | None = None
        self._id_to_vector: dict[int, np.ndarray] = {}
        self._index: faiss.IndexIDMap2 | None = None
        self._last_sweep_at = 0.0
        self._ops_since_sweep = 0
        self._load_index()

    # ── SQLite setup ────────────────────────────────────────────────────

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                created_at  REAL NOT NULL,
                ttl         REAL NOT NULL,
                hit_count   INTEGER DEFAULT 0,
                web_source  INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        return conn

    def _load_index(self) -> None:
        """Load active entries and build ANN index in memory."""
        evicted = self._cleanup_expired_entries()
        rows = self._conn.execute(
            "SELECT id, embedding FROM cache_entries ORDER BY id ASC"
        ).fetchall()
        self._rebuild_index(rows)
        logger.info(
            "Cache loaded %d entries (evicted=%d)",
            len(self._id_to_vector),
            evicted,
        )

    def _create_index(self, dim: int) -> faiss.IndexIDMap2:
        base = faiss.IndexHNSWFlat(dim, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efSearch = self._hnsw_ef_search
        return faiss.IndexIDMap2(base)

    def _rebuild_index(self, rows: list[tuple[int, bytes]]) -> None:
        self._index = None
        self._id_to_vector = {}
        self._vector_dim = None

        vectors: list[np.ndarray] = []
        ids: list[int] = []
        skipped = 0
        for row_id, emb_blob in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32).copy()
            if vec.size == 0:
                skipped += 1
                continue

            if self._vector_dim is None:
                self._vector_dim = int(vec.shape[0])
            if vec.shape[0] != self._vector_dim:
                skipped += 1
                continue

            normalized = self._normalize(vec)
            vectors.append(normalized)
            ids.append(int(row_id))
            self._id_to_vector[int(row_id)] = normalized

        if not vectors or self._vector_dim is None:
            return

        index = self._create_index(self._vector_dim)
        matrix = np.asarray(vectors, dtype=np.float32)
        id_array = np.asarray(ids, dtype=np.int64)
        index.add_with_ids(matrix, id_array)
        self._index = index
        if skipped:
            logger.warning(
                "Cache skipped %d entries due to embedding mismatch", skipped
            )

    # ── Public API ──────────────────────────────────────────────────────

    def get(self, question: str) -> str | None:
        """Look up a cached answer and return only the answer payload."""
        return self.lookup(question).answer

    def lookup(self, question: str) -> CacheLookupResult:
        """Look up a cached answer and return detailed lookup metadata."""
        started = time.perf_counter()
        with self._lock:
            evicted_count = self._maybe_sweep()
            if self._index is None or self._index.ntotal == 0:
                return CacheLookupResult(
                    answer=None,
                    outcome="miss",
                    lookup_ms=(time.perf_counter() - started) * 1000,
                    candidates_scanned=0,
                    similarity_top1=None,
                    evicted_count=evicted_count,
                )

            try:
                q_vec = self._normalize(self._embed(question))
            except Exception:
                logger.warning(
                    "Cache embed failed on lookup; skipping cache", exc_info=True
                )
                return CacheLookupResult(
                    answer=None,
                    outcome="error",
                    lookup_ms=(time.perf_counter() - started) * 1000,
                    candidates_scanned=0,
                    similarity_top1=None,
                    evicted_count=evicted_count,
                )

            if self._vector_dim is None or q_vec.shape[0] != self._vector_dim:
                return CacheLookupResult(
                    answer=None,
                    outcome="miss",
                    lookup_ms=(time.perf_counter() - started) * 1000,
                    candidates_scanned=0,
                    similarity_top1=None,
                    evicted_count=evicted_count,
                )

            k = min(self._top_k, max(1, self._index.ntotal))
            scores, ids = self._index.search(q_vec.reshape(1, -1), k)
            candidate_ids = [
                int(row_id) for row_id in ids[0].tolist() if int(row_id) >= 0
            ]
            candidate_scores = [
                float(score) for score in scores[0].tolist()[: len(candidate_ids)]
            ]

            similarity_top1 = candidate_scores[0] if candidate_scores else None
            if similarity_top1 is None or similarity_top1 < self._similarity_threshold:
                return CacheLookupResult(
                    answer=None,
                    outcome="miss",
                    lookup_ms=(time.perf_counter() - started) * 1000,
                    candidates_scanned=len(candidate_ids),
                    similarity_top1=similarity_top1,
                    evicted_count=evicted_count,
                )

            now = time.time()
            for row_id, score in zip(candidate_ids, candidate_scores):
                if score < self._similarity_threshold:
                    break
                row = self._conn.execute(
                    "SELECT answer, created_at, ttl FROM cache_entries WHERE id = ?",
                    (row_id,),
                ).fetchone()
                if row is None:
                    continue

                answer, created_at, ttl = row
                if now > created_at + ttl:
                    continue

                self._conn.execute(
                    "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?",
                    (row_id,),
                )
                self._conn.commit()
                return CacheLookupResult(
                    answer=answer,
                    outcome="hit",
                    lookup_ms=(time.perf_counter() - started) * 1000,
                    candidates_scanned=len(candidate_ids),
                    similarity_top1=similarity_top1,
                    evicted_count=evicted_count,
                )

            return CacheLookupResult(
                answer=None,
                outcome="miss",
                lookup_ms=(time.perf_counter() - started) * 1000,
                candidates_scanned=len(candidate_ids),
                similarity_top1=similarity_top1,
                evicted_count=evicted_count,
            )

    def put(
        self,
        question: str,
        answer: str,
        web_source: bool = False,
    ) -> None:
        """Store a Q&A pair in the cache."""
        with self._lock:
            try:
                q_vec = self._normalize(self._embed(question))
            except Exception:
                logger.warning(
                    "Cache embed failed on put; skipping cache", exc_info=True
                )
                return
            ttl = _TTL_WEB if web_source else _TTL_STATIC

            blob = q_vec.astype(np.float32).tobytes()
            cur = self._conn.execute(
                """INSERT INTO cache_entries
                   (question, answer, embedding, created_at, ttl, web_source)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (question, answer, blob, time.time(), ttl, int(web_source)),
            )
            self._conn.commit()

            row_id = int(cur.lastrowid)
            if self._vector_dim is None:
                self._vector_dim = int(q_vec.shape[0])
                self._index = self._create_index(self._vector_dim)

            if self._vector_dim == int(q_vec.shape[0]) and self._index is not None:
                self._index.add_with_ids(
                    q_vec.reshape(1, -1).astype(np.float32),
                    np.array([row_id], dtype=np.int64),
                )
                self._id_to_vector[row_id] = q_vec
            else:
                logger.warning(
                    "Skipping ANN insert for row %d due to embedding dimension mismatch",
                    row_id,
                )

            self._maybe_sweep(force=False)

    def clear(self) -> None:
        """Remove all cache entries."""
        with self._lock:
            self._conn.execute("DELETE FROM cache_entries")
            self._conn.commit()
            self._index = None
            self._id_to_vector.clear()
            self._vector_dim = None
            self._ops_since_sweep = 0
            self._last_sweep_at = time.time()

    # ── Internals ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        result = run_with_retry(
            "semantic_cache.embed_query",
            lambda: self._embeddings.embed_query(text),
            logger_instance=logger,
        )
        return np.array(result, dtype=np.float32)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        normalized = vec.astype(np.float32, copy=True)
        norm = float(np.linalg.norm(normalized))
        if norm <= 0:
            return normalized
        normalized /= norm
        return normalized

    def _maybe_sweep(self, force: bool = False) -> int:
        self._ops_since_sweep += 1
        now = time.time()
        should_sweep = force
        should_sweep = should_sweep or (
            now - self._last_sweep_at >= self._sweep_interval_seconds
        )
        should_sweep = should_sweep or (self._ops_since_sweep >= self._sweep_every_ops)
        if not should_sweep:
            return 0

        evicted = self._cleanup_expired_entries()
        pruned = self._enforce_max_entries()
        if evicted or pruned or force:
            rows = self._conn.execute(
                "SELECT id, embedding FROM cache_entries ORDER BY id ASC"
            ).fetchall()
            self._rebuild_index(rows)

        self._last_sweep_at = now
        self._ops_since_sweep = 0
        return evicted + pruned

    def _cleanup_expired_entries(self) -> int:
        now = time.time()
        expired_rows = self._conn.execute(
            "SELECT id FROM cache_entries WHERE created_at + ttl <= ?",
            (now,),
        ).fetchall()
        if not expired_rows:
            return 0

        self._conn.execute(
            "DELETE FROM cache_entries WHERE created_at + ttl <= ?", (now,)
        )
        self._conn.commit()
        return len(expired_rows)

    def _enforce_max_entries(self) -> int:
        count_row = self._conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()
        total = int(count_row[0] if count_row else 0)
        overflow = total - self._max_entries
        if overflow <= 0:
            return 0

        self._conn.execute(
            """
            DELETE FROM cache_entries
            WHERE id IN (
                SELECT id FROM cache_entries
                ORDER BY hit_count ASC, created_at ASC
                LIMIT ?
            )
            """,
            (overflow,),
        )
        self._conn.commit()
        return overflow
