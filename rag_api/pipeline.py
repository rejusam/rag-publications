"""Retrieve relevant chunks, then ask the LLM to answer from them only."""

from __future__ import annotations

import os
import re
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
- Cite papers using short names like (John et al., 2024) — derive \
the author and year from the filename in the [Source:] tags.
- Be concise but thorough.
- Keep answers under 150 words for readability.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


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


def format_docs(docs: Sequence[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        sections.append(f"[Source: {source}]\n{doc.page_content}")
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
            {"context": format_docs(docs), "question": question}
        )
        text = strip_reasoning(raw)
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
