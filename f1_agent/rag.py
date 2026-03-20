"""
RAG pipeline using LangChain to index and retrieve
F1 Technical Regulations from PDF files (2025 and 2026).
"""

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

PDF_FILES: dict[int, str] = {
    2025: "fia_2025_f1_technical_regulations.pdf",
    2026: "fia_2026_f1_technical_regulations.pdf",
}

EMBEDDING_MODEL: str = cast(
    str,
    config(
        "GEMINI_EMBEDDING_MODEL",
        default="models/gemini-embedding-2-preview",
        cast=str,
    ),
)


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


def _vector_store_dir(year: int) -> Path:
    return VECTOR_STORE_DIR / str(year)


def build_vector_store(year: int = 2026) -> FAISS:
    """Load PDF, split into chunks and build FAISS vector store."""
    pdf_filename = PDF_FILES.get(year)
    if not pdf_filename:
        raise ValueError(f"No PDF mapped for year {year}.")

    pdf_path = DOCS_DIR / pdf_filename

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}.\n"
            f"Download the FIA {year} F1 Technical Regulations and place it at:\n"
            f"  docs/{pdf_filename}"
        )

    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1_000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    store_dir = _vector_store_dir(year)
    store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(store_dir))

    return vector_store


def load_vector_store(year: int = 2026) -> FAISS:
    """Load existing vector store from disk."""
    embeddings = _get_embeddings()
    store_dir = _vector_store_dir(year)
    return FAISS.load_local(
        str(store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_vector_store(year: int = 2026) -> FAISS:
    """Return vector store, building it if it doesn't exist yet."""
    store_dir = _vector_store_dir(year)
    if store_dir.exists() and any(store_dir.iterdir()):
        return load_vector_store(year)
    return build_vector_store(year)


def retrieve_context(query: str, k: int = 5, year: int = 2026) -> str:
    """Retrieve the most relevant chunks for a given query."""
    vector_store = get_vector_store(year)
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
