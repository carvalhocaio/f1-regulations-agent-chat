"""RAG-based regulations search tool."""

from __future__ import annotations

import logging
import os

from f1_agent.rag import hybrid_search
from f1_agent.tools_validation import (
    _MAX_QUERY_LEN,
    _normalize_non_empty_text,
    _tool_error,
)

logger = logging.getLogger(__name__)

_RAG_BACKEND_ENV = "F1_RAG_BACKEND"
_RAG_BACKEND_LOCAL = "local"


def search_regulations(query: str) -> dict:
    """Search the FIA 2026 F1 Regulations for relevant information.

    Uses hybrid search (FAISS semantic + BM25 keyword) with reciprocal rank
    fusion for better results, especially when searching for specific article
    numbers or technical terms.

    Args:
        query: The search query about F1 regulations.
    """
    normalized_query = _normalize_non_empty_text(
        value=query,
        field_name="query",
        max_len=_MAX_QUERY_LEN,
    )
    if normalized_query is None:
        return _tool_error(
            tool_name="search_regulations",
            code="INVALID_ARGUMENT",
            message="search_regulations requires non-empty string `query`.",
            details={"field": "query", "expected": "non-empty string"},
        )

    backend = _selected_rag_backend()
    if backend != _RAG_BACKEND_LOCAL:
        logger.warning(
            "Unsupported F1_RAG_BACKEND=%r; using local regulations search", backend
        )

    results = _search_regulations_local(normalized_query, k=5)

    if not results:
        return {"status": "no_results", "message": "No relevant regulations found."}

    chunks = []
    for doc in results:
        chunk = {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "section": doc.metadata.get("section", "unknown"),
        }
        article = doc.metadata.get("article")
        if article:
            chunk["article"] = article
        chunks.append(chunk)

    return {"status": "success", "results": chunks}


def _selected_rag_backend() -> str:
    return os.environ.get(_RAG_BACKEND_ENV, _RAG_BACKEND_LOCAL).strip().lower()


def _search_regulations_local(query: str, k: int = 5):
    return hybrid_search(query, k=k)
