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

    contexts = _extract_contexts(response)
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
    # Newer SDK signature (vertexai.rag):
    # retrieval_query(text=..., rag_resources=[RagResource(...)],
    #                 rag_retrieval_config=RagRetrievalConfig(...))
    try:
        rag_resources = [rag_module.RagResource(rag_corpus=corpus_name)]
        retrieval_filter = (
            rag_module.Filter(vector_distance_threshold=threshold)
            if threshold is not None
            else None
        )
        rag_retrieval_config = rag_module.RagRetrievalConfig(
            top_k=similarity_top_k,
            filter=retrieval_filter,
        )
        return rag_module.retrieval_query(
            text=query,
            rag_resources=rag_resources,
            rag_retrieval_config=rag_retrieval_config,
        )
    except (TypeError, AttributeError):
        pass

    # Legacy/alternate signature fallback.
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


def _extract_contexts(response: Any) -> list[Any]:
    top = getattr(response, "contexts", None)
    if top is None:
        if isinstance(response, dict):
            top = response.get("contexts")
        if top is None:
            return []

    nested = getattr(top, "contexts", None)
    if nested is not None:
        return list(nested or [])

    if isinstance(top, list):
        return top
    if isinstance(top, tuple):
        return list(top)
    if isinstance(top, dict):
        inner = top.get("contexts")
        if isinstance(inner, list):
            return inner
    try:
        return list(top)
    except TypeError:
        return []


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
