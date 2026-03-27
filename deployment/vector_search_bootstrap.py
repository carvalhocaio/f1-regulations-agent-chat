"""Bootstrap Vertex Vector Search collection from local regulation chunks.

Creates a collection (if missing) and ingests chunks from the local FAISS docstore
using the same embedding model currently configured for local retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from typing import Any

from google.cloud import vectorsearch_v1beta as vectorsearch
from google.protobuf import struct_pb2
from langchain_core.documents import Document

from f1_agent.rag import _get_embeddings, get_vector_store


def _slug(text: str, max_len: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", text.strip().lower()).strip("-")
    return value[:max_len] or "doc"


def _to_struct(value: dict[str, Any]) -> struct_pb2.Struct:
    out = struct_pb2.Struct()
    out.update(value)
    return out


def _chunk_id(doc: Document) -> str:
    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", "unknown"))
    article = str(doc.metadata.get("article", ""))
    base = f"{source}|{page}|{article}|{doc.page_content[:120]}"
    digest = hashlib.sha1(base.encode("utf-8"), usedforsecurity=False).hexdigest()[:20]
    return f"f1-{digest}"


def _iter_documents(max_docs: int) -> list[Document]:
    vs = get_vector_store()
    docs: list[Document] = []
    for doc_id in vs.index_to_docstore_id.values():
        found = vs.docstore.search(doc_id)
        if isinstance(found, Document):
            docs.append(found)
            if len(docs) >= max_docs:
                break
    return docs


def _ensure_collection(
    client: vectorsearch.VectorSearchServiceClient,
    *,
    project_id: str,
    location: str,
    collection_id: str,
    dimensions: int,
) -> str:
    name = f"projects/{project_id}/locations/{location}/collections/{collection_id}"
    try:
        client.get_collection(name=name)
        return name
    except Exception:
        pass

    parent = f"projects/{project_id}/locations/{location}"
    data_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "source": {"type": "string"},
            "section": {"type": "string"},
            "page": {"type": "string"},
            "article": {"type": "string"},
        },
    }

    collection = vectorsearch.Collection(
        display_name=f"F1 Regulations {_slug(collection_id, max_len=24)}",
        description="F1 regulations chunk embeddings for retrieval benchmark",
        data_schema=_to_struct(data_schema),
        vector_schema={
            "embedding": vectorsearch.VectorField(
                dense_vector=vectorsearch.DenseVectorField(dimensions=dimensions)
            )
        },
    )
    op = client.create_collection(
        request=vectorsearch.CreateCollectionRequest(
            parent=parent,
            collection_id=collection_id,
            collection=collection,
        )
    )
    op.result(timeout=600)
    return name


def _ingest_documents(
    data_client: vectorsearch.DataObjectServiceClient,
    *,
    parent: str,
    docs: list[Document],
) -> tuple[int, int]:
    embeddings = _get_embeddings()
    created = 0
    skipped = 0

    for doc in docs:
        doc_id = _chunk_id(doc)
        vector = embeddings.embed_query(doc.page_content)
        payload = {
            "content": doc.page_content,
            "source": str(doc.metadata.get("source", "unknown")),
            "section": str(doc.metadata.get("section", "unknown")),
            "page": str(doc.metadata.get("page", "unknown")),
            "article": str(doc.metadata.get("article", "")),
        }

        request = vectorsearch.CreateDataObjectRequest(
            parent=parent,
            data_object_id=doc_id,
            data_object=vectorsearch.DataObject(
                data=_to_struct(payload),
                vectors={
                    "embedding": vectorsearch.Vector(
                        dense=vectorsearch.DenseVector(
                            values=[float(v) for v in vector]
                        )
                    )
                },
            ),
        )

        try:
            data_client.create_data_object(request=request)
            created += 1
        except Exception as exc:
            if "already exists" in str(exc).lower():
                skipped += 1
                continue
            raise

    return created, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Vector Search collection and ingest local chunks"
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--collection-id", default="f1-regulations-benchmark")
    parser.add_argument("--max-docs", type=int, default=600)
    args = parser.parse_args()

    if args.max_docs < 1:
        raise ValueError("--max-docs must be >= 1")

    dims = len(_get_embeddings().embed_query("f1 regulations bootstrap"))
    vector_client = vectorsearch.VectorSearchServiceClient()
    data_client = vectorsearch.DataObjectServiceClient()

    collection_name = _ensure_collection(
        vector_client,
        project_id=args.project_id,
        location=args.location,
        collection_id=args.collection_id,
        dimensions=dims,
    )

    docs = _iter_documents(max_docs=args.max_docs)
    created, skipped = _ingest_documents(data_client, parent=collection_name, docs=docs)

    print(f"collection={collection_name}")
    print(f"docs_selected={len(docs)} created={created} skipped_existing={skipped}")


if __name__ == "__main__":
    main()
