"""Vertex AI Vector Search retrieval adapter.

This module is intentionally defensive: if Vector Search client/resources are not
available, callers can fallback to other retrieval backends.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_VS_PARENT_ENV = "F1_VECTOR_SEARCH_PARENT"
_VS_SEARCH_FIELD_ENV = "F1_VECTOR_SEARCH_FIELD"
_VS_TOP_K_ENV = "F1_VECTOR_SEARCH_TOP_K"
_VS_OUTPUT_FIELDS_ENV = "F1_VECTOR_SEARCH_OUTPUT_FIELDS"

_DEFAULT_SEARCH_FIELD = "embedding"
_DEFAULT_OUTPUT_FIELDS = {
    "data_fields": "*",
    "vector_fields": "",
    "metadata_fields": "*",
}


def vector_search_retrieve(query: str, k: int = 5) -> list[Document]:
    parent = os.environ.get(_VS_PARENT_ENV, "").strip()
    if not parent:
        raise ValueError(f"Missing {_VS_PARENT_ENV} for Vector Search retrieval")

    search_field = os.environ.get(_VS_SEARCH_FIELD_ENV, _DEFAULT_SEARCH_FIELD).strip()
    top_k = _int_env(_VS_TOP_K_ENV, default=k)

    query_vector = _embed_query(query)
    if not query_vector:
        return []

    client, vectorsearch = _load_vectorsearch_client()
    request = vectorsearch.SearchDataObjectsRequest(
        parent=parent,
        vector_search=vectorsearch.VectorSearch(
            search_field=search_field,
            vector={"values": query_vector},
            top_k=top_k,
            output_fields=_output_fields(vectorsearch),
        ),
    )

    response = client.search_data_objects(request=request)
    docs: list[Document] = []
    for result in list(getattr(response, "results", []) or []):
        data_object = getattr(result, "data_object", None)
        if data_object is None:
            continue

        metadata = _extract_metadata(data_object)
        text = _extract_text(data_object)
        if not text:
            continue
        docs.append(Document(page_content=text, metadata=metadata))

    return docs[:k]


def _embed_query(query: str) -> list[float]:
    from f1_agent.rag import _get_embeddings

    embeddings = _get_embeddings()
    vector = embeddings.embed_query(query)
    if not vector:
        return []
    return [float(v) for v in vector]


def _load_vectorsearch_client():
    from google.cloud import vectorsearch_v1beta

    return (
        vectorsearch_v1beta.DataObjectSearchServiceClient(),
        vectorsearch_v1beta,
    )


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, Mapping):
        return dict(attrs)
    return {}


def _extract_text(data_object: Any) -> str:
    data_fields = _as_dict(_field(data_object, "data_fields"))
    if not data_fields:
        data_fields = _as_dict(_field(data_object, "data"))
    metadata_fields = _as_dict(_field(data_object, "metadata_fields"))
    for key in ("content", "text", "chunk_text", "snippet"):
        value = data_fields.get(key)
        if isinstance(value, str) and value.strip():
            return value
        value = metadata_fields.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_metadata(data_object: Any) -> dict[str, Any]:
    data_fields = _as_dict(_field(data_object, "data_fields"))
    if not data_fields:
        data_fields = _as_dict(_field(data_object, "data"))
    metadata_fields = _as_dict(_field(data_object, "metadata_fields"))

    def _pick(*keys: str, fallback: str = "unknown") -> str:
        for key in keys:
            value = metadata_fields.get(key)
            if value is None:
                value = data_fields.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return fallback

    metadata = {
        "source": _pick("source", "source_uri", fallback="vector_search"),
        "page": _pick("page", "page_number"),
        "section": _pick("section"),
    }
    article = _pick("article", fallback="")
    if article:
        metadata["article"] = article
    return metadata


def _output_fields(vectorsearch: Any) -> Any:
    raw = os.environ.get(_VS_OUTPUT_FIELDS_ENV, "").strip()
    if not raw:
        return vectorsearch.OutputFields(**_DEFAULT_OUTPUT_FIELDS)

    fields = [part.strip() for part in raw.split(",") if part.strip()]
    if not fields:
        return vectorsearch.OutputFields(**_DEFAULT_OUTPUT_FIELDS)

    selected = {
        "data_fields": "*" if "data_fields" in fields else "",
        "vector_fields": "*" if "vector_fields" in fields else "",
        "metadata_fields": "*" if "metadata_fields" in fields else "",
    }
    if not any(selected.values()):
        selected = _DEFAULT_OUTPUT_FIELDS
    return vectorsearch.OutputFields(**selected)


def _field(data_object: Any, name: str) -> Any:
    value = getattr(data_object, name, None)
    if value is not None:
        return value
    if isinstance(data_object, Mapping):
        return data_object.get(name)
    return None


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid int env %s=%r; using %s", name, raw, default)
        return default
