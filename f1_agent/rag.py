"""
RAG pipeline using LangChain to index and retrieve
FIA 2026 F1 Regulations from PDF files (Sections A–F).

Improvements over the baseline:
- Larger chunk size (1500 chars) to keep full articles together
- Article-aware separators that respect regulation structure
- BM25 keyword index built alongside FAISS for hybrid search
- Reciprocal rank fusion to combine semantic + keyword results
"""

import logging
import re
import time
from pathlib import Path
from typing import cast

from decouple import config
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
from rank_bm25 import BM25Okapi

from f1_agent.env_utils import get_package_dir

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent.parent / "docs"


def _resolve_vector_store_dir(base_dir: Path) -> Path:
    """Resolve vector_store directory from either flat or nested package layouts."""
    if (base_dir / "index.faiss").exists() and (base_dir / "index.pkl").exists():
        return base_dir

    nested_dir = base_dir / "vector_store"
    if (nested_dir / "index.faiss").exists() and (nested_dir / "index.pkl").exists():
        return nested_dir

    return base_dir


# When deployed, vector_store is an installed package in site-packages.
# Locally, it's a sibling directory of f1_agent.
try:
    import vector_store as _vs_pkg

    _vector_store_base_dir = get_package_dir(_vs_pkg)
except ImportError:
    _vector_store_base_dir = Path(__file__).parent.parent / "vector_store"

VECTOR_STORE_DIR = _resolve_vector_store_dir(_vector_store_base_dir)

EMBEDDING_MODEL: str = cast(
    str,
    config(
        "GEMINI_EMBEDDING_MODEL",
        default="models/gemini-embedding-2-preview",
        cast=str,
    ),
)

BUILD_EMBED_BATCH_SIZE: int = cast(
    int,
    config("F1_BUILD_EMBED_BATCH_SIZE", default=20, cast=int),
)
BUILD_EMBED_MAX_RETRIES: int = cast(
    int,
    config("F1_BUILD_EMBED_MAX_RETRIES", default=6, cast=int),
)
BUILD_EMBED_BASE_DELAY_S: float = cast(
    float,
    config("F1_BUILD_EMBED_BASE_DELAY_S", default=2.0, cast=float),
)
BUILD_EMBED_MAX_DELAY_S: float = cast(
    float,
    config("F1_BUILD_EMBED_MAX_DELAY_S", default=90.0, cast=float),
)
BUILD_EMBED_SLEEP_BETWEEN_BATCHES_S: float = cast(
    float,
    config("F1_BUILD_EMBED_SLEEP_BETWEEN_BATCHES_S", default=0.35, cast=float),
)

SECTION_PATTERN = re.compile(
    r"section[_ ]+([a-f])[_ \[]+(.+?)[\]_ ]*-[_ ]*iss",
    re.IGNORECASE,
)


def _extract_section(filename: str) -> str:
    """Extract a human-readable section label from a PDF filename."""
    match = SECTION_PATTERN.search(filename)
    if match:
        letter = match.group(1).upper()
        description = match.group(2).strip().replace("_", " ").title()
        # Clean up extra spaces and dashes
        description = re.sub(r"\s*-\s*", " — ", description)
        return f"Section {letter} — {description}"
    return "Unknown Section"


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key: str = cast(str, config("GEMINI_API_KEY", default="", cast=str))
    if not api_key:
        api_key = cast(str, config("GOOGLE_API_KEY", default="", cast=str))
    if not api_key:
        raise ValueError(
            "API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY in .env."
        )

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SecretStr(api_key),
    )


def _extract_retry_delay_seconds(error_text: str) -> float | None:
    retry_matches = [
        re.search(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text),
        re.search(r"'retryDelay':\s*'([0-9]+)s'", error_text),
    ]
    for match in retry_matches:
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _is_rate_limited_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text


class _ResilientEmbeddings:
    def __init__(self, base: GoogleGenerativeAIEmbeddings):
        self._base = base

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batch_size = max(1, BUILD_EMBED_BATCH_SIZE)
        max_retries = max(1, BUILD_EMBED_MAX_RETRIES)
        base_delay = max(0.1, BUILD_EMBED_BASE_DELAY_S)
        max_delay = max(base_delay, BUILD_EMBED_MAX_DELAY_S)
        per_batch_sleep = max(0.0, BUILD_EMBED_SLEEP_BETWEEN_BATCHES_S)

        embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            attempt = 0
            while True:
                try:
                    batch_embeddings = self._base.embed_documents(
                        batch,
                        batch_size=batch_size,
                    )
                    embeddings.extend(batch_embeddings)
                    if per_batch_sleep > 0:
                        time.sleep(per_batch_sleep)
                    break
                except Exception as exc:
                    attempt += 1
                    if not _is_rate_limited_error(exc) or attempt >= max_retries:
                        raise

                    retry_after = _extract_retry_delay_seconds(str(exc))
                    exp_backoff = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    sleep_s = max(retry_after or 0.0, exp_backoff)
                    logger.warning(
                        "Embedding rate limited while building index; "
                        "retrying batch %d-%d in %.2fs (attempt %d/%d)",
                        i,
                        min(i + len(batch), len(texts)),
                        sleep_s,
                        attempt,
                        max_retries,
                    )
                    time.sleep(sleep_s)

        return embeddings


# Regex to extract article numbers from regulation text (e.g. "3.2.1", "12.4")
_ARTICLE_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){1,3})\b")


def _extract_article(text: str) -> str:
    """Extract the first article number found in chunk text."""
    match = _ARTICLE_RE.search(text)
    return match.group(1) if match else ""


def build_vector_store() -> FAISS:
    """Load all PDF files from docs/, split into chunks and build FAISS vector store."""
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCS_DIR}.\n"
            "Place the FIA 2026 F1 Regulations PDFs (Sections A–F) in the docs/ "
            "directory."
        )

    all_chunks = []
    # Article-aware separators: split on article boundaries first, then paragraphs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_500,
        chunk_overlap=200,
        separators=[
            "\nArticle ",  # Article boundaries
            "\nArt. ",  # Abbreviated article references
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            " ",  # Words (fallback)
        ],
    )

    for pdf_path in pdf_files:
        section = _extract_section(pdf_path.name)
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()

        for doc in documents:
            doc.metadata["section"] = section

        chunks = splitter.split_documents(documents)

        # Enrich chunks with article metadata
        for chunk in chunks:
            article = _extract_article(chunk.page_content)
            if article:
                chunk.metadata["article"] = article

        all_chunks.extend(chunks)

    embeddings = _ResilientEmbeddings(_get_embeddings())
    vector_store = FAISS.from_documents(all_chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))

    return vector_store


def load_vector_store() -> FAISS:
    """Load existing vector store from disk."""
    embeddings = _get_embeddings()
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_vector_store() -> FAISS:
    """Return vector store, building it if it doesn't exist yet."""
    if VECTOR_STORE_DIR.exists() and any(VECTOR_STORE_DIR.iterdir()):
        return load_vector_store()
    return build_vector_store()


def retrieve_context(query: str, k: int = 5) -> str:
    """Retrieve the most relevant chunks for a given query."""
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── BM25 keyword index ──────────────────────────────────────────────────

_bm25_index: BM25Okapi | None = None
_bm25_docs: list[Document] = []


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer for BM25."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _build_bm25_index(documents: list[Document]) -> BM25Okapi:
    """Build a BM25 index from a list of documents."""
    corpus = [_tokenize(doc.page_content) for doc in documents]
    return BM25Okapi(corpus)


def _get_bm25() -> tuple[BM25Okapi, list[Document]]:
    """Return the BM25 index and document list, building if needed.

    The BM25 index is built from the same FAISS vector store documents.
    """
    global _bm25_index, _bm25_docs

    if _bm25_index is not None:
        return _bm25_index, _bm25_docs

    vs = get_vector_store()
    # Extract all documents from the FAISS store
    all_docs = []
    docstore = vs.docstore
    for doc_id in vs.index_to_docstore_id.values():
        doc = docstore.search(doc_id)
        if isinstance(doc, Document):
            all_docs.append(doc)

    _bm25_docs = all_docs
    _bm25_index = _build_bm25_index(all_docs)
    return _bm25_index, _bm25_docs


def bm25_search(query: str, k: int = 10) -> list[Document]:
    """Search the BM25 keyword index and return top-k documents."""
    bm25, docs = _get_bm25()
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_indices if scores[i] > 0]


def hybrid_search(query: str, k: int = 5) -> list[Document]:
    """Combine FAISS semantic search and BM25 keyword search via reciprocal rank fusion.

    Returns the top-k documents after fusing both result lists.
    """
    # Get more candidates from each source, then fuse
    vs = get_vector_store()
    semantic_results = vs.similarity_search(query, k=k * 2)
    keyword_results = bm25_search(query, k=k * 2)

    # Reciprocal rank fusion (RRF) with k=60 (standard constant)
    rrf_k = 60
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(semantic_results):
        doc_id = f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{doc.page_content[:50]}"
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(keyword_results):
        doc_id = f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{doc.page_content[:50]}"
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[doc_id] = doc

    # Sort by fused score and return top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [doc_map[doc_id] for doc_id, _ in ranked]
