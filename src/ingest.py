"""Ingest research papers into a ChromaDB vector store.

This module handles the full ingestion pipeline:
    1. Load PDFs from data/papers/
    2. Split into overlapping chunks
    3. Embed with a local Ollama model
    4. Persist to ChromaDB on disk

Run directly:
    python -m src.ingest
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    PAPERS_DIR,
    SEPARATORS,
)


# ─── Step 1: Load ────────────────────────────────────────────────────


def load_papers(papers_dir: Path = PAPERS_DIR) -> list[Document]:
    """Load all PDFs from the papers directory.

    Each page is tagged with ``source_file`` metadata so answers
    can cite which paper they drew from.

    Args:
        papers_dir: Path to directory containing PDF files.

    Returns:
        Flat list of Document objects (one per page).

    Raises:
        FileNotFoundError: If papers_dir does not exist.
        ValueError: If no PDFs are found.
    """
    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")

    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(
            f"No PDF files found in {papers_dir}. "
            "Download your papers into data/papers/ first."
        )

    docs: list[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.stem
            page.metadata["source_path"] = str(pdf_path.name)
        docs.extend(pages)
        print(f"  ✓ {pdf_path.name} ({len(pages)} pages)")

    print(f"\nLoaded {len(docs)} pages from {len(pdf_paths)} papers")
    return docs


# ─── Step 2: Chunk ───────────────────────────────────────────────────


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks.

    Uses RecursiveCharacterTextSplitter which tries to split on
    paragraph boundaries first, then sentences, then words.

    Args:
        docs: List of full-page Document objects.

    Returns:
        List of smaller Document chunks with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ─── Step 3: Embed + Store ───────────────────────────────────────────


def create_vector_store(chunks: list[Document]) -> Chroma:
    """Embed document chunks and persist to ChromaDB.

    Args:
        chunks: Pre-split Document chunks to embed.

    Returns:
        Initialised Chroma vector store with persisted data.
    """
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # Wipe existing store to avoid duplicates on re-ingest
    if CHROMA_DIR.exists():
        import shutil

        shutil.rmtree(CHROMA_DIR)
        print(f"Cleared existing vector store at {CHROMA_DIR}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Vector store persisted to {CHROMA_DIR} ({len(chunks)} vectors)")
    return vector_store


# ─── Entrypoint ──────────────────────────────────────────────────────


def run_ingestion() -> None:
    """Execute the full ingestion pipeline."""
    print("=" * 60)
    print("RAG Publications — Ingestion Pipeline")
    print("=" * 60)

    print("\n[1/3] Loading papers...")
    docs = load_papers()

    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(docs)

    print("\n[3/3] Embedding and storing...")
    create_vector_store(chunks)

    print("\n✓ Ingestion complete. You can now run the app.")


if __name__ == "__main__":
    try:
        run_ingestion()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
