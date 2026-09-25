"""Map retrieved chunks to citable papers.

Citations shown to users come from this lookup, not from model output,
so a paper can only be cited if it was actually retrieved.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PAPERS_PATH = Path(__file__).resolve().parent.parent / "papers.json"


@dataclass(frozen=True)
class Source:
    title: str
    authors_short: str | None = None
    year: int | None = None
    journal: str | None = None
    doi: str | None = None


def load_papers(path: Path = PAPERS_PATH) -> dict[str, Source]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: Source(**value) for key, value in raw.items()}


def resolve_sources(
    docs: Sequence[Document], papers: Mapping[str, Source]
) -> list[Source]:
    """Unique papers behind the retrieved chunks, in retrieval order."""
    seen: set[str] = set()
    sources: list[Source] = []
    for doc in docs:
        key = doc.metadata.get("source_file")
        if not key or key in seen:
            continue
        seen.add(key)
        source = papers.get(key)
        if source is None:
            logger.warning("No papers.json entry for source_file=%r", key)
            source = Source(title=key)
        sources.append(source)
    return sources
