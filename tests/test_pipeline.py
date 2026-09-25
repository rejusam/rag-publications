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
    normalise_citations,
    strip_markdown,
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Rodents spread it 【John et al., 2024】.",
            "Rodents spread it (John et al., 2024).",
        ),
        (
            "A 【Smith et al., 2020】 and B 【Jones et al., 2021】.",
            "A (Smith et al., 2020) and B (Jones et al., 2021).",
        ),
        ("No brackets here.", "No brackets here."),
        ("Stray open 【 only.", "Stray open ( only."),
        ("Stray close 】 only.", "Stray close ) only."),
    ],
)
def test_normalise_citations(raw, expected):
    assert normalise_citations(raw) == expected


def test_ask_normalises_full_width_citation_brackets():
    llm = FakeListChatModel(responses=["Answer text 【John et al., 2024】."])
    result = Pipeline(RETRIEVER, llm, PAPERS).ask("q?")
    assert result.text == "Answer text (John et al., 2024)."
    assert "【" not in result.text
    assert "】" not in result.text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**Inhalation** of dust", "Inhalation of dust"),
        ("shed by *M. natalensis* rodents", "shed by M. natalensis rodents"),
        ("- first\n- second", "first\nsecond"),
        ("## Heading\ntext", "Heading\ntext"),
        ("2 * 3 = 6", "2 * 3 = 6"),
        ("Plain text.", "Plain text."),
        ("- 0.5 correlation", "- 0.5 correlation"),
        ("- 3 cases were reported", "- 3 cases were reported"),
        ("#1 risk factor", "#1 risk factor"),
        ("-  0.5 corr", "-  0.5 corr"),
        ("-\t2 x", "-\t2 x"),
        ("*  bold item", "bold item"),
    ],
)
def test_strip_markdown(raw, expected):
    assert strip_markdown(raw) == expected


def test_ask_strips_markdown_from_answer():
    llm = FakeListChatModel(responses=["**Bold** answer with *italic* text."])
    result = Pipeline(RETRIEVER, llm, PAPERS).ask("q?")
    assert result.text == "Bold answer with italic text."


def test_format_docs_labels_sources():
    text = format_docs(DOCS, PAPERS)
    assert text.count("[Source: John et al., 2024]") == 2
    assert "[Source: lassa]" not in text
    assert "Rodents shed virus." in text


def test_format_docs_unmapped_key_uses_raw_source_file():
    docs = [Document(page_content="Unmapped text.", metadata={"source_file": "unmapped-key"})]
    text = format_docs(docs, PAPERS)
    assert "[Source: unmapped-key]" in text


def test_format_docs_disambiguates_shared_author_year_labels():
    papers = {
        "john-a": Source(
            title="Travel time and disease transmission across large connected populations",
            authors_short="John et al.",
            year=2024,
        ),
        "john-b": Source(
            title="Modelling Lassa virus dynamics in rodents and human spillover risk",
            authors_short="John et al.",
            year=2024,
        ),
    }
    docs = [
        Document(page_content="A text.", metadata={"source_file": "john-a"}),
        Document(page_content="B text.", metadata={"source_file": "john-b"}),
    ]
    text = format_docs(docs, papers)
    assert "[Source: John et al., 2024 — Travel time and disease transmission across]" in text
    assert "[Source: John et al., 2024 — Modelling Lassa virus dynamics in rodents]" in text


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
    assert "[Source: John et al., 2024]" in seen[0]
    assert "How does Lassa spread?" in seen[0]
    assert "ONLY the provided context" in seen[0]
    assert (
        "Never cite references that appear inside the context text" in seen[0]
    )
    assert "Write in plain prose without Markdown" in seen[0]


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
