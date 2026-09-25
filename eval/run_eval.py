"""Evaluate retrieval (offline) and, optionally, answers (live).

Usage, from the repo root:
    python -m eval.run_eval          # retrieval hit rate only, no API key needed
    python -m eval.run_eval --live   # also records answers for manual refusal review
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from rag_api.config import Settings, load_env_file
from rag_api.pipeline import RETRIEVER_K, build_pipeline, build_retriever
from rag_api.sources import load_papers

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "eval" / "results.md"


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _label(key: str | None, papers) -> str:
    if key is None:
        return "—"
    src = papers.get(key)
    return f"{src.authors_short} ({src.year})" if src else key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    papers = load_papers()
    retriever = build_retriever()

    rows = []
    hits = answerable = 0
    for q in questions:
        retrieved: list[str] = []
        for doc in retriever.invoke(q["question"]):
            key = doc.metadata.get("source_file")
            if key and key not in retrieved:
                retrieved.append(key)
        hit = None
        if q["expected"] is not None:
            answerable += 1
            hit = q["expected"] in retrieved
            hits += hit
        rows.append((q, retrieved, hit))

    answers: dict[str, str] = {}
    if args.live:
        load_env_file(ROOT / ".env")
        pipeline = build_pipeline(Settings.from_env())
        for q in questions:
            answers[q["id"]] = pipeline.ask(q["question"]).text

    lines = [
        "# Evaluation results",
        "",
        f"- Date: {dt.date.today().isoformat()}",
        f"- Commit: {_commit()}",
        f"- Retriever: top-{RETRIEVER_K} similarity",
        f"- Retrieval hit rate (expected paper among retrieved): "
        f"**{hits}/{answerable}** ({hits / answerable:.0%})",
        "",
        "| id | question | expected | retrieved | hit |",
        "|---|---|---|---|---|",
    ]
    for q, retrieved, hit in rows:
        got = ", ".join(_label(k, papers) for k in retrieved)
        mark = "—" if hit is None else ("yes" if hit else "no")
        lines.append(
            f"| {q['id']} | {q['question']} | {_label(q['expected'], papers)} | {got} | {mark} |"
        )
    if answers:
        lines += [
            "",
            "## Answers (live)",
            "",
            "Unanswerable questions (`u*`) should be declined. Mark each by hand.",
            "",
        ]
        for q in questions:
            lines += [f"**{q['id']}. {q['question']}**", "", answers[q["id"]], "",
                      "Reviewer verdict: ______", ""]
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Hit rate {hits}/{answerable}. Wrote {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
