from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langchain_core.documents import Document

from rag_api.sources import Source, load_papers, resolve_sources

ROOT = Path(__file__).resolve().parent.parent

PAPERS = {
    "a": Source(title="Paper A", authors_short="John et al.", year=2024, journal="J", doi="10.1/a"),
    "b": Source(title="Paper B", authors_short="Hayman et al.", year=2022, journal="J", doi="10.1/b"),
}


def doc(source_file):
    meta = {} if source_file is None else {"source_file": source_file}
    return Document(page_content="text", metadata=meta)


def test_real_papers_file_loads():
    papers = load_papers()
    assert len(papers) == 6
    for src in papers.values():
        assert src.title and src.authors_short and src.journal
        assert isinstance(src.year, int)
        assert src.doi.startswith("10.")


def test_resolve_dedupes_and_keeps_retrieval_order():
    out = resolve_sources([doc("b"), doc("a"), doc("b")], PAPERS)
    assert [s.title for s in out] == ["Paper B", "Paper A"]


def test_unmapped_file_falls_back_to_raw_title_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="rag_api.sources"):
        out = resolve_sources([doc("mystery-file")], PAPERS)
    assert out == [Source(title="mystery-file")]
    assert "mystery-file" in caplog.text


def test_docs_without_source_file_are_ignored():
    assert resolve_sources([doc(None), doc("")], PAPERS) == []


def test_every_indexed_paper_is_mapped():
    db = ROOT / "chroma_db_deploy" / "chroma.sqlite3"
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata WHERE key = 'source_file'"
        ).fetchall()
    indexed = {r[0] for r in rows}
    assert indexed, "no source_file metadata found in the vector store"
    missing = indexed - set(load_papers())
    assert not missing, f"papers.json is missing entries for: {sorted(missing)}"
