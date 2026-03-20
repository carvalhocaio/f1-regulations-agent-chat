from f1_agent.rag import get_vector_store


def search_regulations(query: str) -> dict:
    """Search the FIA 2026 F1 Technical Regulations for relevant information.

    Args:
        query: The search query about F1 technical regulations.
    """
    vector_store = get_vector_store(year=2026)
    results = vector_store.similarity_search(query, k=5)

    if not results:
        return {"status": "no_results", "message": "No relevant regulations found."}

    chunks = []
    for doc in results:
        chunks.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
            }
        )

    return {"status": "success", "results": chunks}


def compare_with_previous_year(query: str) -> dict:
    """Search the FIA 2025 F1 Technical Regulations for the same topic,
    so the agent can compare with the 2026 regulations.

    Args:
        query: The search query about F1 technical regulations.
    """
    vector_store = get_vector_store(year=2025)
    results = vector_store.similarity_search(query, k=5)

    if not results:
        return {
            "status": "no_results",
            "message": "No relevant sections found in the 2025 regulations.",
        }

    chunks = []
    for doc in results:
        chunks.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
            }
        )

    return {"status": "success", "year": 2025, "results": chunks}
