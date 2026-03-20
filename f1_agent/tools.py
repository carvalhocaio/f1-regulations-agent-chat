from f1_agent.rag import get_vector_store


def search_regulations(query: str) -> dict:
    """Search the FIA 2026 F1 Regulations for relevant information.

    Args:
        query: The search query about F1 regulations.
    """
    vector_store = get_vector_store()
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
                "section": doc.metadata.get("section", "unknown"),
            }
        )

    return {"status": "success", "results": chunks}
