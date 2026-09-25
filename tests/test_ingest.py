"""Tests for the ingestion pipeline.

These tests validate the loading and chunking logic without
requiring Ollama or ChromaDB to be running (those are integration
tests you run manually).
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_ollama", reason="local Ollama stack not installed")

from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.ingest import chunk_documents, load_papers


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def sample_docs() -> list[Document]:
    """Create sample documents mimicking PDF page output."""
    return [
        Document(
            page_content="This is a study on pandemic spread. " * 50,
            metadata={"source_file": "john_2024_connectivity", "page": 0},
        ),
        Document(
            page_content="Lassa virus dynamics in West Africa. " * 50,
            metadata={"source_file": "john_2024_lassa", "page": 0},
        ),
    ]


# ─── Tests: load_papers ─────────────────────────────────────────────


def test_load_papers_missing_dir():
    """Should raise FileNotFoundError for nonexistent directory."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_papers(Path("/nonexistent/path"))


def test_load_papers_empty_dir(tmp_path: Path):
    """Should raise ValueError when no PDFs are found."""
    with pytest.raises(ValueError, match="No PDF files found"):
        load_papers(tmp_path)


# ─── Tests: chunk_documents ──────────────────────────────────────────


def test_chunk_documents_produces_chunks(sample_docs: list[Document]):
    """Chunking should produce more chunks than input documents."""
    chunks = chunk_documents(sample_docs)
    assert len(chunks) > len(sample_docs)


def test_chunk_documents_preserves_metadata(sample_docs: list[Document]):
    """Every chunk should retain the source_file metadata."""
    chunks = chunk_documents(sample_docs)
    for chunk in chunks:
        assert "source_file" in chunk.metadata
        assert chunk.metadata["source_file"] in [
            "john_2024_connectivity",
            "john_2024_lassa",
        ]


def test_chunk_documents_respects_size(sample_docs: list[Document]):
    """No chunk should exceed the configured chunk size (with tolerance)."""
    from src.config import CHUNK_SIZE

    chunks = chunk_documents(sample_docs)
    for chunk in chunks:
        # Allow small overflow from the splitter's boundary logic
        assert len(chunk.page_content) <= CHUNK_SIZE + 50


def test_chunk_documents_empty_input():
    """Empty input should produce empty output, not crash."""
    chunks = chunk_documents([])
    assert chunks == []
