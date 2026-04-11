"""Retrieve relevant document chunks from the vector store.

This module loads the persisted ChromaDB and exposes a LangChain
retriever object for use in the RAG chain.
"""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVER_K,
    SEARCH_TYPE,
)


def _load_vector_store() -> Chroma:
    """Load the persisted ChromaDB vector store.

    Raises:
        FileNotFoundError: If the vector store has not been created yet.
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}. "
            "Run `python -m src.ingest` first."
        )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


def get_retriever(k: int = RETRIEVER_K) -> VectorStoreRetriever:
    """Return a LangChain retriever backed by the persisted vector store.

    Args:
        k: Number of chunks to retrieve per query.

    Returns:
        A VectorStoreRetriever configured for similarity search.
    """
    store = _load_vector_store()
    return store.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": k},
    )


def search(query: str, k: int = RETRIEVER_K) -> list[Document]:
    """Convenience function: run a similarity search and return Documents.

    Useful for debugging retrieval quality outside the full chain.

    Args:
        query: Natural language question.
        k: Number of results.

    Returns:
        List of the k most relevant Document chunks.
    """
    store = _load_vector_store()
    return store.similarity_search(query, k=k)
