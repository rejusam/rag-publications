"""Export the deployed Chroma chunks to JSONL for the Cloudflare Worker.

The PDFs are not in the repo, so the chunk text in chroma_db_deploy is the
source of truth. Opened read-only so the store is never rewritten.

    python -m scripts.export_chunks
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DB = ROOT / "chroma_db_deploy" / "chroma.sqlite3"
OUT = ROOT / "worker" / "data" / "chunks.jsonl"


def export_chunks(db_path: Path = CHROMA_DB, out_path: Path = OUT) -> int:
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            """
            SELECT e.embedding_id,
                   MAX(CASE WHEN m.key = 'chroma:document' THEN m.string_value END),
                   MAX(CASE WHEN m.key = 'source_file' THEN m.string_value END),
                   MAX(CASE WHEN m.key = 'page' THEN m.int_value END)
            FROM embeddings e
            JOIN embedding_metadata m ON m.id = e.id
            GROUP BY e.id
            ORDER BY e.id
            """
        ).fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for embedding_id, text, source_file, page in rows:
            if not text:
                continue
            record = {
                "id": embedding_id,
                "text": text,
                "source_file": source_file,
                "page": page,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    print(f"Wrote {export_chunks()} chunks to {OUT}")
