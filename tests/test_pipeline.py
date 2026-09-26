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
    filter_citations,
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


FILTER_SOURCES = [
    Source(title="Travel time paper", authors_short="John et al.", year=2024),
    Source(title="Land use paper", authors_short="Rulli et al.", year=2025),
    Source(title="Ebola paper", authors_short="Hayman et al.", year=2022),
]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Rodents shed virus (Safronetz et al., 2022; Wozniak et al., 2021).",
            "Rodents shed virus.",
        ),
        (
            "Contact spreads it (Rulli et al., 2025).",
            "Contact spreads it (Rulli et al., 2025).",
        ),
        (
            "Mixed (Rulli et al., 2025; Lo Iacono et al., 2015).",
            "Mixed (Rulli et al., 2025).",
        ),
        (
            "(John et al., 2024 — Modelling Lassa virus dynamics in West)",
            "(John et al., 2024 — Modelling Lassa virus dynamics in West)",
        ),
        ("during the (2014–2016 outbreak)", "during the (2014–2016 outbreak)"),
        ("No parentheses here.", "No parentheses here."),
    ],
)
def test_filter_citations(raw, expected):
    assert filter_citations(raw, FILTER_SOURCES) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("(Smith et al., 2019 – title)", ""),
        ("(Smith et al., 2019, p. 4)", ""),
        ("(Smith et al.,2019)", ""),
        ("(Smith et al., 2019, 2020)", ""),
        ("((Smith et al., 2019))", ""),
        ("(see Rulli et al., 2025)", "(see Rulli et al., 2025)"),
        ("(e.g. Rulli et al., 2025)", "(e.g. Rulli et al., 2025)"),
        ("(in 2019)", "(in 2019)"),
        ("(March 2020)", "(March 2020)"),
        ("(since 2018)", "(since 2018)"),
        ("(Mastomys natalensis, 2019)", "(Mastomys natalensis, 2019)"),
        ("(R0 ≈ 1.5; 2019)", "(R0 ≈ 1.5; 2019)"),
        ("Values fell to .5", "Values fell to .5"),
        ("Wait ... really , yes.", "Wait ... really , yes."),
        (
            "Rats matter. (Lo Iacono et al., 2015) showed that rats matter. End.",
            "Rats matter. Showed that rats matter. End.",
        ),
        ("(Smith et al., 2019; 2020)", ""),
        ("(Hayman et al., 2022)", "(Hayman et al., 2022)"),
        ("(Hayman, 2022)", ""),
        ("A\n(Smith et al., 2019).", "A\n."),
    ],
)
def test_filter_citations_strict_grammar(raw, expected):
    assert filter_citations(raw, FILTER_SOURCES) == expected


def test_filter_citations_long_whitespace_run_is_fast():
    import time

    raw = "(a" + " " * 20000 + "x 2019)"
    start = time.perf_counter()
    filter_citations(raw, FILTER_SOURCES)
    assert time.perf_counter() - start < 0.2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "First. (Smith et al., 2019) showed R0 was 1.5 in Sierra Leone. "
            "Second sentence stays.",
            "First. Showed R0 was 1.5 in Sierra Leone. Second sentence stays.",
        ),
        (
            'First. (Smith, 2019) found "rats." Next one. Another. Last.',
            'First. Found "rats." Next one. Another. Last.',
        ),
        (
            "(Smith, 2019) showed it (John et al., 2024). Keep.",
            "Showed it (John et al., 2024). Keep.",
        ),
        (
            "First. (Smith et al., 2019) found rats\n\nNew paragraph here. Stays.",
            "First. Found rats\n\nNew paragraph here. Stays.",
        ),
        (
            "First. (Smith et al., 2019) showed Dr. Smith was right. Next.",
            "First. Showed Dr. Smith was right. Next.",
        ),
        (
            "A.\r\n(Smith et al., 2019) found rats.\r\nNext.",
            "A.\r\nFound rats.\r\nNext.",
        ),
        ("(a;b 2019)", "(a;b 2019)"),
    ],
)
def test_filter_citations_plain_removal_capitalises_sentence_start(raw, expected):
    assert filter_citations(raw, FILTER_SOURCES) == expected


def test_filter_citations_many_dropped_sentence_starts_is_fast():
    import time

    raw = ("Sentence. (Smith et al., 2019) rats spread it. " * 5000)
    start = time.perf_counter()
    filter_citations(raw, FILTER_SOURCES)
    assert time.perf_counter() - start < 0.3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("shed virus (Fichet‑Calvet et al., 2014).", "shed virus."),
        ("shed virus (Fichet‐Calvet et al., 2014).", "shed virus."),
        ("(Fichet-Calvet et al., 2014)", ""),
    ],
)
def test_filter_citations_drops_unicode_hyphen_surname(raw, expected):
    assert filter_citations(raw, FILTER_SOURCES) == expected


def test_filter_citations_keeps_unicode_hyphen_surname_matching_ascii_source():
    sources = [*FILTER_SOURCES, Source(
        title="Fichet-Calvet paper", authors_short="Fichet-Calvet et al.", year=2014,
    )]
    raw = "shed virus (Fichet‑Calvet et al., 2014)."
    assert filter_citations(raw, sources) == raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("shed (Müller et al., 2019).", "shed."),
        ("shed (Gómez et al., 2020).", "shed."),
        ("shed (Ødegaard et al., 2018).", "shed."),
        ("shed (van der Berg et al., 2017).", "shed."),
        ("shed (in 2019).", "shed (in 2019)."),
        ("shed (de facto, 2019).", "shed (de facto, 2019)."),
    ],
)
def test_filter_citations_unicode_and_particle_surnames(raw, expected):
    assert filter_citations(raw, FILTER_SOURCES) == expected


@pytest.mark.parametrize(
    ("authors_short", "year", "raw"),
    [
        ("Gómez et al.", 2020, "shed (Gómez et al., 2020)."),
        ("van der Berg et al.", 2017, "shed (van der Berg et al., 2017)."),
    ],
)
def test_filter_citations_keeps_in_corpus_unicode_and_particle_surnames(
    authors_short, year, raw
):
    sources = [*FILTER_SOURCES, Source(
        title="Extra paper", authors_short=authors_short, year=year,
    )]
    assert filter_citations(raw, sources) == raw


def test_ask_drops_out_of_corpus_citations_keeps_in_corpus():
    llm = FakeListChatModel(
        responses=["Spread happens via contact (John et al., 2024; Lo Iacono et al., 2015)."]
    )
    result = Pipeline(RETRIEVER, llm, PAPERS).ask("q?")
    assert result.text == "Spread happens via contact (John et al., 2024)."
    assert "Lo Iacono" not in result.text


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


def test_model_list_raises_last_failure_when_all_fail():
    class FirstError(Exception):
        pass

    class LastError(Exception):
        pass

    def raise_first(_):
        raise FirstError()

    def raise_last(_):
        raise LastError()

    pipeline = Pipeline(
        RETRIEVER, [RunnableLambda(raise_first), RunnableLambda(raise_last)], PAPERS
    )
    with pytest.raises(LastError):
        pipeline.ask("q?")


def test_model_list_uses_fallback_when_primary_fails():
    llms = [RunnableLambda(failing), FakeListChatModel(responses=["From fallback."])]
    assert Pipeline(RETRIEVER, llms, PAPERS).ask("q?").text == "From fallback."
