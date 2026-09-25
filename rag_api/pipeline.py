"""Retrieve relevant chunks, then ask the LLM to answer from them only."""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from rag_api.config import Settings
from rag_api.sources import Source, load_papers, resolve_sources

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db_deploy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_K = 4

SYSTEM_TEMPLATE = """\
You are a research assistant for Dr Reju Sam John, a computational \
epidemiologist and data scientist based in Auckland, New Zealand.

INSTRUCTIONS:
- Answer the question using ONLY the provided context from his \
published peer-reviewed papers.
- If the context does not contain enough information to answer, \
say so honestly — do not hallucinate.
- Cite only the papers named in the [Source: ...] tags, using that \
label in brackets, e.g. (John et al., 2024).
- Never cite references that appear inside the context text (for \
example other authors the papers themselves cite).
- Write in plain prose without Markdown: no asterisks, bullet lists, \
headings or bold text. Use short paragraphs separated by a blank \
line if needed.
- Do not end with a list of sources or quotations; the sources are \
shown to the reader separately.
- Be concise but thorough.
- Keep answers under 150 words for readability.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FULLWIDTH_CITATION = re.compile(r"【(.*?)】")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_ITALIC = re.compile(r"\*(\S(?:[^\n*]*\S)?)\*")
_LEADING_MARKER = re.compile(
    r"(?m)^[ \t]*(?:[-*•][ \t]+(?![ \t\d])|#{1,6}\s+)"
)


class EmptyAnswerError(RuntimeError):
    """The model produced no usable answer text."""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]


def strip_reasoning(text: str) -> str:
    """Remove any reasoning markup so only the final answer reaches users."""
    text = _THINK_BLOCK.sub("", text)
    start = text.lower().find("<think>")
    if start != -1:
        text = text[:start]
    return text.strip()


def normalise_citations(text: str) -> str:
    """Replace full-width citation brackets some models emit with plain parens.

    Some models (e.g. gpt-oss) sometimes emit citations wrapped in 【...】
    instead of the (Author et al., Year) form requested in the prompt. Fix
    this deterministically rather than relying on further prompt tuning.
    """
    text = _FULLWIDTH_CITATION.sub(lambda match: f"({match.group(1)})", text)
    return text.replace("【", "(").replace("】", ")")


def strip_markdown(text: str) -> str:
    """Strip common Markdown so answers read as plain prose in the widget.

    Small and regex-based on purpose: this only has to undo the handful
    of Markdown patterns models sometimes emit despite the prompt asking
    for plain prose, not parse arbitrary Markdown.
    """
    def _unwrap(match: re.Match[str]) -> str:
        return match.group(1) if match.group(1) is not None else match.group(2)

    text = _BOLD.sub(_unwrap, text)
    text = _ITALIC.sub(lambda m: m.group(1), text)
    return _LEADING_MARKER.sub("", text)


def _citation_label(
    source_file: str | None,
    papers: Mapping[str, Source],
    shared_author_year_counts: Mapping[tuple[str, int], int],
) -> str:
    source = papers.get(source_file) if source_file else None
    if source is not None and source.authors_short and source.year:
        label = f"{source.authors_short}, {source.year}"
        if shared_author_year_counts[(source.authors_short, source.year)] > 1:
            title_words = " ".join(source.title.split()[:6])
            return f"{label} — {title_words}"
        return label
    return source_file or "unknown"


def format_docs(docs: Sequence[Document], papers: Mapping[str, Source]) -> str:
    shared_author_year_counts = Counter(
        (source.authors_short, source.year)
        for source in papers.values()
        if source.authors_short and source.year
    )
    sections = []
    for doc in docs:
        label = _citation_label(
            doc.metadata.get("source_file"), papers, shared_author_year_counts
        )
        sections.append(f"[Source: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


class Pipeline:
    def __init__(
        self, retriever: Runnable, llm: Runnable, papers: Mapping[str, Source]
    ) -> None:
        self._retriever = retriever
        self._generate = (
            ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
            | llm
            | StrOutputParser()
        )
        self._papers = papers

    def ask(self, question: str) -> Answer:
        docs = self._retriever.invoke(question)
        raw = self._generate.invoke(
            {"context": format_docs(docs, self._papers), "question": question}
        )
        text = strip_reasoning(raw)
        text = normalise_citations(text)
        text = strip_markdown(text)
        if not text:
            raise EmptyAnswerError("model returned no answer text")
        return Answer(text=text, sources=resolve_sources(docs, self._papers))


def build_retriever(chroma_dir: Path = CHROMA_DIR) -> Runnable:
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_community.vectorstores import Chroma

    store = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=FastEmbedEmbeddings(model_name=EMBEDDING_MODEL),
    )
    return store.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
    )


def build_pipeline(settings: Settings) -> Pipeline:
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set")

    from langchain_groq import ChatGroq

    def groq(model: str) -> ChatGroq:
        return ChatGroq(
            model=model,
            temperature=0.1,
            timeout=settings.llm_timeout_s,
            max_retries=1,
        )

    llm = groq(settings.groq_model).with_fallbacks(
        [groq(settings.groq_fallback_model)]
    )
    return Pipeline(build_retriever(), llm, load_papers())
