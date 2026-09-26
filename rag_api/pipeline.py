"""Retrieve relevant chunks, then ask the LLM to answer from them only."""

from __future__ import annotations

import os
import re
import unicodedata
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
_INNER_GROUP = re.compile(r"\(([^()]*)\)")
_BARE_YEAR = re.compile(r"^\d{4}[a-z]?$")
_MONTH_NAMES = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    }
)
_MONTH_WORD = re.compile(r"[^\W\d_]+")

# Lowercase name particles that may precede a surname ("van der Berg").
_PARTICLES = (
    "van", "von", "der", "den", "de", "del", "della", "da", "di", "du",
    "le", "la", "dos", "das", "ter", "ten", "'d", "’d", "bin", "al-",
)
_PARTICLE_SET = frozenset(_PARTICLES)
_PARTICLE = "(?:" + "|".join(re.escape(p) for p in _PARTICLES) + r")\s+"
# A surname word is Unicode letters plus apostrophes and hyphens (no digits
# or underscores). ‐/‑ are the hyphen and non-breaking hyphen gpt-oss
# frequently emits in place of ASCII "-" (e.g. "Fichet‑Calvet"). Python's
# `re` can't express "uppercase letter", so the capitalised first letter
# is checked in code by `_is_surname_author`.
_SURNAME_WORD = r"[^\W\d_](?:[^\W\d_]|['’\-‐‑])+"
_SURNAME = (
    rf"(?:{_PARTICLE}){{0,3}}{_SURNAME_WORD}(?:\s+{_SURNAME_WORD})?"
)
_AUTHOR = rf"{_SURNAME}(?:\s+et\s+al\.?|\s+(?:and|&)\s+{_SURNAME})?"
_YEAR = r"\d{4}[a-z]?"
_CITATION_ENTRY = re.compile(
    rf"^(?:(?:see|e\.g\.,?|cf\.)\s+)?"
    rf"(?P<author>{_AUTHOR}),?[ \t]{{0,3}}(?P<year>{_YEAR})"
    rf"(?:,[ \t]{{0,3}}\d{{4}}[a-z]?)*"
    rf"(?:,[ \t]{{0,3}}(?:p|pp)\.[ \t]{{0,3}}[\d\-–]+"
    rf"|[ \t]{{0,3}}[—–-][ \t]{{0,3}}.+)?$"
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


def _normalise_author(author: str) -> str:
    # Fold the hyphen look-alikes gpt-oss uses (‐, ‑) to ASCII "-"
    # so a hyphenated surname matches regardless of which one was written;
    # this only affects the matching key, never the answer text itself.
    author = unicodedata.normalize("NFC", author)
    author = author.replace("‐", "-").replace("‑", "-")
    return " ".join(author.replace(".", "").replace(",", "").split()).casefold()


_ET_AL_SUFFIX = re.compile(r"\s+et\s+al\.?$")
_AUTHOR_JOINER = re.compile(r"\s+(?:and|&)\s+")


def _is_surname_author(author: str) -> bool:
    """Check each surname has at most 3 leading particles then 1–2 words
    whose first letter is uppercase ("van der Berg", "Lo Iacono", "Ødegaard").

    Rejects lowercase phrases the regex alone would accept ("in 2019",
    "de facto, 2019") so they are kept verbatim as non-citations.
    """
    author = _ET_AL_SUFFIX.sub("", author)
    for surname in _AUTHOR_JOINER.split(author):
        words = surname.split()
        particles = 0
        while particles < 3 and len(words) > 1 and words[0] in _PARTICLE_SET:
            words = words[1:]
            particles += 1
        if not 1 <= len(words) <= 2 or not all(w[0].isupper() for w in words):
            return False
    return True


def _has_month_token(author: str) -> bool:
    return any(w.lower() in _MONTH_NAMES for w in _MONTH_WORD.findall(author))


def _split_group_entries(
    content: str, known: set[tuple[str, int]]
) -> tuple[list[str], bool]:
    """Return (kept entries, whether anything in the group was dropped).

    When nothing was dropped, the caller must leave the original group text
    byte-for-byte unchanged rather than re-joining `kept` — re-joining can
    silently change formatting a citation-free group never asked to have
    changed (e.g. "(a;b 2019)" becoming "(a; b 2019)").
    """
    if not content.strip():
        return [], True
    if not re.search(r"\d{4}", content):
        return [content], False

    kept: list[str] = []
    dropped_any = False
    prev_dropped_citation = False
    for raw_entry in content.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        # NFC for matching only, so a decomposed "Mu\u0308ller" still parses;
        # a kept entry is appended exactly as written.
        match = _CITATION_ENTRY.match(unicodedata.normalize("NFC", entry))
        if (
            match
            and _is_surname_author(match.group("author"))
            and not _has_month_token(match.group("author"))
        ):
            author = _normalise_author(match.group("author"))
            year = int(match.group("year")[:4])
            key = (author, year)
            if key in known:
                kept.append(entry)
                prev_dropped_citation = False
            else:
                dropped_any = True
                prev_dropped_citation = True
            continue
        if prev_dropped_citation and _BARE_YEAR.match(entry):
            dropped_any = True
            continue  # orphan year left behind by a dropped citation
        kept.append(entry)
        prev_dropped_citation = False
    return kept, dropped_any


def _run_filter_pass(text: str, known: set[tuple[str, int]]) -> str:
    out: list[str] = []
    last_end = 0
    n = len(text)
    for match in _INNER_GROUP.finditer(text):
        start, end = match.span()
        kept, dropped_any = _split_group_entries(match.group(1), known)
        prefix_text = text[last_end:start]

        if not dropped_any:
            # Nothing removed: leave the group exactly as it was written.
            out.append(prefix_text)
            out.append(match.group(0))
            last_end = end
            continue

        if kept:
            out.append(prefix_text)
            out.append(f"({'; '.join(kept)})")
            last_end = end
            continue

        # Every entry in the group was dropped: remove it via plain local
        # tidy only. An earlier version tried to detect and remove the
        # whole broken sentence that followed, but reliable sentence-end
        # detection is its own hard problem (abbreviations, quotes, nested
        # citations, quadratic-time scans) and not worth the risk here —
        # so the "sentence" itself is left alone; only the one boundary
        # letter that would otherwise start mid-word gets capitalised.
        so_far = "".join(out) + prefix_text
        stripped = so_far.rstrip(" ")
        is_sentence_start = not stripped or stripped[-1] in ".!?\n"

        left_trim = 1 if so_far and so_far[-1] == " " else 0
        out.append(prefix_text[: len(prefix_text) - left_trim])
        # No preceding space survived (either nothing precedes the group at
        # all, or it was preceded by a newline or other non-space
        # boundary): absorb one following space too, so the removal never
        # leaves a stray leading space or an orphaned separator behind.
        # Otherwise, only absorb a following space when it precedes
        # `.`/`,`/`;` — a plain word-separating space is left in place.
        right_extra = 1 if (
            end < n and text[end] == " "
            and (left_trim == 0 or (end + 1 < n and text[end + 1] in ".,;"))
        ) else 0
        last_end = end + right_extra

        if is_sentence_start:
            peek = last_end
            if peek < n and text[peek] == " ":
                peek += 1
            if peek < n and text[peek].islower():
                out.append(text[last_end:peek])
                out.append(text[peek].upper())
                last_end = peek + 1

    out.append(text[last_end:])
    return "".join(out)


def filter_citations(text: str, sources: Sequence[Source]) -> str:
    """Drop parenthesised citations that don't name a retrieved paper.

    Models sometimes cite references that only appear inside the
    retrieved paper text (works those papers themselves cite), not the
    papers actually retrieved. Enforce "only cite retrieved papers"
    deterministically instead of relying solely on the prompt.

    Runs to a fixed point so a nested group left empty by one pass
    (e.g. "((Smith et al., 2019))") is cleaned up by the next.
    """
    known = {
        (_normalise_author(source.authors_short), source.year)
        for source in sources
        if source.authors_short and source.year
    }
    previous = None
    while previous != text:
        previous = text
        text = _run_filter_pass(text, known)
    return text


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
        self,
        retriever: Runnable,
        llms: Runnable | Sequence[Runnable],
        papers: Mapping[str, Source],
    ) -> None:
        """`llms` is one model, or models to try in order (primary first)."""
        if isinstance(llms, Runnable):
            llms = [llms]
        if not llms:
            raise ValueError("at least one model is required")
        self._retriever = retriever
        prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
        self._generators = [prompt | llm | StrOutputParser() for llm in llms]
        self._papers = papers

    def _generate(self, inputs: Mapping[str, str]) -> str:
        """Try each model in turn; if all fail, raise the last failure.

        An explicit loop rather than LangChain's `with_fallbacks`, which
        re-raises the first model's error: the caller maps the error to a
        status code, and it should reflect the model that failed last.
        """
        last_error: Exception | None = None
        for generate in self._generators:
            try:
                return generate.invoke(inputs)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def ask(self, question: str) -> Answer:
        docs = self._retriever.invoke(question)
        raw = self._generate(
            {"context": format_docs(docs, self._papers), "question": question}
        )
        text = strip_reasoning(raw)
        text = normalise_citations(text)
        text = strip_markdown(text)
        sources = resolve_sources(docs, self._papers)
        text = filter_citations(text, sources)
        if not text:
            raise EmptyAnswerError("model returned no answer text")
        return Answer(text=text, sources=sources)


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

    llms = [groq(settings.groq_model), groq(settings.groq_fallback_model)]
    return Pipeline(build_retriever(), llms, load_papers())
