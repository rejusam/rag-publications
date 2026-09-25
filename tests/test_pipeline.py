from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from rag_api.config import Settings
from rag_api.pipeline import (
    EmptyAnswerError,
    Pipeline,
    build_pipeline,
    format_docs,
    strip_reasoning,
)
from rag_api.sources import Source

PAPERS = {"lassa": Source(title="Lassa paper", authors_short="John et al.", year=2024)}
DOCS = [
    Document(page_content="Rodents shed virus.", metadata={"source_file": "lassa"}),
    Document(page_content="More on rodents.", metadata={"source_file": "lassa"}),
]
RETRIEVER = RunnableLambda(lambda _q: DOCS)


def failing(_):
    raise RuntimeError("primary down")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Plain answer.", "Plain answer."),
        ("<think>secret steps</think>Final.", "Final."),
        ("<THINK>\nmulti\nline\n</THINK>\n  Final.  ", "Final."),
        ("Answer. <think>trailing unterminated", "Answer."),
        ("<think>only reasoning", ""),
    ],
)
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected


def test_format_docs_labels_sources():
    text = format_docs(DOCS)
    assert text.count("[Source: lassa]") == 2
    assert "Rodents shed virus." in text


def test_ask_returns_clean_answer_and_sources():
    llm = FakeListChatModel(responses=["<think>hmm</think>Via rodent contact."])
    result = Pipeline(RETRIEVER, llm, PAPERS).ask("How does Lassa spread?")
    assert result.text == "Via rodent contact."
    assert result.sources == [PAPERS["lassa"]]


def test_prompt_contains_context_and_question():
    seen = []

    def record(prompt_value):
        seen.append(prompt_value.to_string())
        return "ok"

    Pipeline(RETRIEVER, RunnableLambda(record), PAPERS).ask("How does Lassa spread?")
    assert "[Source: lassa]" in seen[0]
    assert "How does Lassa spread?" in seen[0]
    assert "ONLY the provided context" in seen[0]


def test_fallback_model_answers_when_primary_fails():
    llm = RunnableLambda(failing).with_fallbacks(
        [FakeListChatModel(responses=["From fallback."])]
    )
    assert Pipeline(RETRIEVER, llm, PAPERS).ask("q?").text == "From fallback."


def test_reasoning_only_output_raises_empty_answer():
    llm = FakeListChatModel(responses=["<think>never finishes"])
    with pytest.raises(EmptyAnswerError):
        Pipeline(RETRIEVER, llm, PAPERS).ask("q?")


def test_build_pipeline_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_pipeline(Settings())
