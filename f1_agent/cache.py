"""
Semantic answer cache backed by FAISS + SQLite.

Embeds user questions with the same Gemini embedding model used for RAG,
stores them in a small FAISS index, and returns cached answers when a
new question is semantically close enough (cosine similarity > threshold).

Storage layout (next to the main vector_store):
    f1_cache/
        cache_index.faiss   — question embeddings
        cache_index.pkl     — FAISS metadata
        cache.db            — SQLite with full Q&A + TTL
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import cast

import numpy as np
from decouple import config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "f1_cache"

# Similarity threshold — 1.0 = identical, lower = more lenient
_SIMILARITY_THRESHOLD = 0.92

# TTLs in seconds
_TTL_STATIC = 30 * 24 * 3600  # 30 days  (historical DB + regulations)
_TTL_WEB = 24 * 3600  # 24 hours (google search results)


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key: str = cast(str, config("GEMINI_API_KEY", default="", cast=str))
    if not api_key:
        api_key = cast(str, config("GOOGLE_API_KEY", default="", cast=str))
    if not api_key:
        raise ValueError("API key required for cache embeddings.")

    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        api_key=SecretStr(api_key),
    )


class SemanticCache:
    """Question-level semantic cache with FAISS similarity search."""

    def __init__(self, cache_dir: Path | None = None):
        self._dir = cache_dir or _CACHE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self._dir / "cache.db"
        self._embeddings = _get_embeddings()
        self._conn = self._init_db()
        self._vectors: list[np.ndarray] = []
        self._ids: list[int] = []
        self._load_vectors()

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

    def _load_vectors(self) -> None:
        """Load all non-expired embeddings into memory."""
        now = time.time()
        rows = self._conn.execute(
            "SELECT id, embedding FROM cache_entries WHERE created_at + ttl > ?",
            (now,),
        ).fetchall()

        self._vectors = []
        self._ids = []
        for row_id, emb_blob in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            self._vectors.append(vec)
            self._ids.append(row_id)

        logger.info("Cache loaded %d entries", len(self._ids))

    # ── Public API ──────────────────────────────────────────────────────

    def get(self, question: str) -> str | None:
        """Look up a cached answer for a semantically similar question.

        Returns the answer string on cache hit, or None on miss.
        """
        if not self._vectors:
            return None

        q_vec = self._embed(question)

        # Compute cosine similarities
        matrix = np.stack(self._vectors)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        similarities = np.dot(matrix, q_vec) / norms

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim < _SIMILARITY_THRESHOLD:
            return None

        row_id = self._ids[best_idx]

        # Check TTL and return
        now = time.time()
        row = self._conn.execute(
            "SELECT answer, created_at, ttl FROM cache_entries WHERE id = ?",
            (row_id,),
        ).fetchone()

        if row is None:
            return None

        answer, created_at, ttl = row
        if now > created_at + ttl:
            # Expired — evict
            self._evict(row_id, best_idx)
            return None

        # Bump hit count
        self._conn.execute(
            "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?",
            (row_id,),
        )
        self._conn.commit()

        return answer

    def put(
        self,
        question: str,
        answer: str,
        web_source: bool = False,
    ) -> None:
        """Store a Q&A pair in the cache."""
        q_vec = self._embed(question)
        ttl = _TTL_WEB if web_source else _TTL_STATIC

        blob = q_vec.tobytes()
        cur = self._conn.execute(
            """INSERT INTO cache_entries
               (question, answer, embedding, created_at, ttl, web_source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (question, answer, blob, time.time(), ttl, int(web_source)),
        )
        self._conn.commit()

        self._vectors.append(q_vec)
        self._ids.append(cur.lastrowid)

    def clear(self) -> None:
        """Remove all cache entries."""
        self._conn.execute("DELETE FROM cache_entries")
        self._conn.commit()
        self._vectors.clear()
        self._ids.clear()

    # ── Internals ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        result = self._embeddings.embed_query(text)
        return np.array(result, dtype=np.float32)

    def _evict(self, row_id: int, vec_idx: int) -> None:
        self._conn.execute("DELETE FROM cache_entries WHERE id = ?", (row_id,))
        self._conn.commit()
        self._vectors.pop(vec_idx)
        self._ids.pop(vec_idx)
