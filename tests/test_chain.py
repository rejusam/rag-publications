"""Tests for the RAG chain module.

These test the pure functions (format_docs, prompt template) without
requiring a running LLM or vector store.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.chain import SYSTEM_TEMPLATE, format_docs


def test_format_docs_single():
    """Single document should be formatted with source tag."""
    docs = [
        Document(
            page_content="Pandemic spread findings.",
            metadata={"source_file": "john_2024_connectivity"},
        )
    ]
    result = format_docs(docs)
    assert "[Source: john_2024_connectivity]" in result
    assert "Pandemic spread findings." in result


def test_format_docs_multiple():
    """Multiple documents should be separated by dividers."""
    docs = [
        Document(page_content="First.", metadata={"source_file": "paper_a"}),
        Document(page_content="Second.", metadata={"source_file": "paper_b"}),
    ]
    result = format_docs(docs)
    assert result.count("---") >= 1
    assert "[Source: paper_a]" in result
    assert "[Source: paper_b]" in result


def test_format_docs_missing_metadata():
    """Documents without source_file should use 'unknown'."""
    docs = [Document(page_content="No metadata.", metadata={})]
    result = format_docs(docs)
    assert "[Source: unknown]" in result


def test_format_docs_empty():
    """Empty list should return empty string."""
    result = format_docs([])
    assert result == ""


def test_system_template_has_placeholders():
    """Prompt template must contain both required placeholders."""
    assert "{context}" in SYSTEM_TEMPLATE
    assert "{question}" in SYSTEM_TEMPLATE
