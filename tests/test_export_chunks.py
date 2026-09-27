from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.export_chunks import export_chunks

ROOT = Path(__file__).resolve().parent.parent


def _fake_chroma(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, embedding_id TEXT)")
        conn.execute(
            "CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INTEGER)"
        )
        conn.executemany("INSERT INTO embeddings VALUES (?, ?)", [(1, "uuid-1"), (2, "uuid-2")])
        conn.executemany(
            "INSERT INTO embedding_metadata VALUES (?, ?, ?, ?)",
            [
                (1, "chroma:document", "First chunk.", None),
                (1, "source_file", "paper-a", None),
                (1, "page", None, 3),
                (2, "chroma:document", "Second chunk.", None),
                (2, "source_file", "paper-b", None),
            ],
        )


def test_export_writes_one_line_per_chunk(tmp_path):
    db = tmp_path / "chroma.sqlite3"
    _fake_chroma(db)
    out = tmp_path / "chunks.jsonl"
    assert export_chunks(db, out) == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"id": "uuid-1", "text": "First chunk.", "source_file": "paper-a", "page": 3},
        {"id": "uuid-2", "text": "Second chunk.", "source_file": "paper-b", "page": None},
    ]


def test_export_real_store_has_779_chunks(tmp_path):
    out = tmp_path / "chunks.jsonl"
    assert export_chunks(ROOT / "chroma_db_deploy" / "chroma.sqlite3", out) == 779
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert first["text"] and first["source_file"]
    assert len(first["id"].encode()) <= 64
