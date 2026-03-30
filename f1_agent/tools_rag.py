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
_RAG_BACKEND_VERTEX = "vertex"
_RAG_BACKEND_VECTOR_SEARCH = "vector_search"
_RAG_BACKEND_AUTO = "auto"


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

    if backend == _RAG_BACKEND_LOCAL:
        results = _search_regulations_local(normalized_query, k=5)
    elif backend == _RAG_BACKEND_VERTEX:
        results = _search_regulations_vertex(normalized_query, k=5)
        if not results:
            logger.warning(
                "Vertex RAG returned no results; falling back to local search"
            )
            results = _search_regulations_local(normalized_query, k=5)
    elif backend == _RAG_BACKEND_VECTOR_SEARCH:
        results = _search_regulations_vector_search(normalized_query, k=5)
        if not results:
            logger.warning(
                "Vector Search returned no results; falling back to local search"
            )
            results = _search_regulations_local(normalized_query, k=5)
    else:
        for candidate in (
            _search_regulations_vector_search,
            _search_regulations_vertex,
            _search_regulations_local,
        ):
            results = candidate(normalized_query, k=5)
            if results:
                break

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
    raw = os.environ.get(_RAG_BACKEND_ENV, _RAG_BACKEND_AUTO).strip().lower()
    if raw in {
        _RAG_BACKEND_LOCAL,
        _RAG_BACKEND_VERTEX,
        _RAG_BACKEND_VECTOR_SEARCH,
        _RAG_BACKEND_AUTO,
    }:
        return raw
    return _RAG_BACKEND_AUTO


def _search_regulations_local(query: str, k: int = 5):
    return hybrid_search(query, k=k)


def _search_regulations_vertex(query: str, k: int = 5):
    try:
        from f1_agent.rag_vertex import vertex_hybrid_search

        return vertex_hybrid_search(query, k=k)
    except Exception:
        logger.warning(
            "Vertex RAG unavailable; falling back to local search", exc_info=True
        )
        return []


def _search_regulations_vector_search(query: str, k: int = 5):
    try:
        from f1_agent.rag_vector_search import vector_search_retrieve

        return vector_search_retrieve(query, k=k)
    except Exception:
        logger.warning(
            "Vector Search unavailable; falling back to next backend", exc_info=True
        )
        return []
