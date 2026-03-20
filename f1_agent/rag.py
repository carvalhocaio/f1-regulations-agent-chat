"""
RAG pipeline using LangChain to index and retrieve
FIA 2026 F1 Regulations from PDF files (Sections A–F).
"""

import re
from pathlib import Path
from typing import cast

from decouple import config
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

DOCS_DIR = Path(__file__).parent.parent / "docs"
VECTOR_STORE_DIR = Path(__file__).parent.parent / "vector_store"

EMBEDDING_MODEL: str = cast(
    str,
    config(
        "GEMINI_EMBEDDING_MODEL",
        default="models/gemini-embedding-2-preview",
        cast=str,
    ),
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
    api_key: str = cast(str, config("GOOGLE_API_KEY", default="", cast=str))
    if not api_key:
        api_key = cast(str, config("GEMINI_API_KEY", default="", cast=str))
    if not api_key:
        raise ValueError(
            "API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env."
        )

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SecretStr(api_key),
    )


def build_vector_store() -> FAISS:
    """Load all PDF files from docs/, split into chunks and build FAISS vector store."""
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCS_DIR}.\n"
            "Place the FIA 2026 F1 Regulations PDFs (Sections A–F) in the docs/ directory."
        )

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )

    for pdf_path in pdf_files:
        section = _extract_section(pdf_path.name)
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()

        for doc in documents:
            doc.metadata["section"] = section

        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)

    embeddings = _get_embeddings()
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
