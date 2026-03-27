import logging
import os
from typing import Any

from langchain_core.documents import Document

from f1_agent.resilience import run_with_retry

logger = logging.getLogger(__name__)

_RAG_CORPUS_ENV = "F1_RAG_CORPUS"
_RAG_PROJECT_ENV = "F1_RAG_PROJECT_ID"
_RAG_LOCATION_ENV = "F1_RAG_LOCATION"
_RAG_TOP_K_ENV = "F1_RAG_TOP_K"
_RAG_DISTANCE_THRESHOLD_ENV = "F1_RAG_VECTOR_DISTANCE_THRESHOLD"


def vertex_hybrid_search(query: str, k: int = 5) -> list[Document]:
    corpus_name = os.environ.get(_RAG_CORPUS_ENV, "").strip()
    if not corpus_name:
        raise ValueError(f"Missing {_RAG_CORPUS_ENV} for Vertex RAG retrieval")

    rag_module, vertexai_module = _load_rag_module()
    _maybe_vertex_init(vertexai_module)

    similarity_top_k = _int_env(_RAG_TOP_K_ENV, default=k)
    threshold = _float_env(_RAG_DISTANCE_THRESHOLD_ENV)

    response = run_with_retry(
        "rag_vertex.retrieval_query",
        lambda: _call_retrieval_query(
            rag_module=rag_module,
            corpus_name=corpus_name,
            query=query,
            similarity_top_k=similarity_top_k,
            threshold=threshold,
        ),
        logger_instance=logger,
    )

    contexts = list(getattr(response, "contexts", []) or [])
    docs: list[Document] = []
    for ctx in contexts:
        text = _context_text(ctx)
        if not text:
            continue

        metadata = {
            "source": _first_non_empty(_context_field(ctx, "source_uri"), "vertex_rag"),
            "page": _first_non_empty(_context_field(ctx, "page_number"), "unknown"),
            "section": _first_non_empty(_context_field(ctx, "section"), "unknown"),
        }
        docs.append(Document(page_content=text, metadata=metadata))

    return docs[:k]


def _load_rag_module():
    import vertexai

    try:
        from vertexai import rag

        return rag, vertexai
    except Exception:
        from vertexai.preview import rag

        return rag, vertexai


def _maybe_vertex_init(vertexai_module: Any) -> None:
    project = os.environ.get(_RAG_PROJECT_ENV, "").strip()
    location = os.environ.get(_RAG_LOCATION_ENV, "").strip()

    if project and location:
        vertexai_module.init(project=project, location=location)
    elif project:
        vertexai_module.init(project=project)
    elif location:
        vertexai_module.init(location=location)


def _call_retrieval_query(
    rag_module: Any,
    corpus_name: str,
    query: str,
    similarity_top_k: int,
    threshold: float | None,
):
    kwargs = {
        "corpus_names": [corpus_name],
        "text": query,
        "similarity_top_k": similarity_top_k,
    }

    if threshold is not None:
        kwargs["vector_distance_threshold"] = threshold

    try:
        return rag_module.retrieval_query(**kwargs)
    except TypeError:
        kwargs.pop("vector_distance_threshold", None)
        return rag_module.retrieval_query(**kwargs)


def _context_text(context: Any) -> str:
    candidate = _context_field(context, "text")
    if candidate:
        return str(candidate)

    candidate = _context_field(context, "chunk_text")
    if candidate:
        return str(candidate)

    candidate = _context_field(context, "content")
    if candidate:
        return str(candidate)

    return ""


def _context_field(context: Any, name: str) -> Any:
    value = getattr(context, name, None)
    if value is not None:
        return value

    if isinstance(context, dict):
        return context.get(name)

    return None


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _float_env(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _first_non_empty(value: Any, fallback: str) -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback
