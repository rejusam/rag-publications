from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from rag_api.app import create_app
from rag_api.config import Settings
from rag_api.pipeline import Pipeline
from rag_api.sources import Source

PAPERS = {
    "lassa": Source(
        title="Lassa paper", authors_short="John et al.", year=2024,
        journal="J. R. Soc. Interface", doi="10.1098/rsif.2024.0106",
    )
}
DOCS = [
    Document(page_content="Rodents shed virus.", metadata={"source_file": "lassa"}),
    Document(page_content="Unmapped.", metadata={"source_file": "unmapped-file"}),
]
SECRET = "internal detail gsk_must_never_leak"


class UpstreamError(Exception):
    def __init__(self, status_code=None):
        super().__init__(SECRET)
        self.status_code = status_code


def raising(status_code=None):
    def _raise(_):
        raise UpstreamError(status_code)
    return RunnableLambda(_raise)


def make_app(llm, settings=None):
    retriever = RunnableLambda(lambda _q: DOCS)
    return create_app(
        settings or Settings(),
        pipeline_factory=lambda _s: Pipeline(retriever, llm, PAPERS),
    )


def ok_llm(text="Via rodent contact."):
    return FakeListChatModel(responses=[text])


def test_ask_returns_answer_and_sources():
    with TestClient(make_app(ok_llm())) as client:
        r = client.post("/ask", json={"question": "How does Lassa spread?"})
    assert r.status_code == 200
    assert r.json() == {
        "answer": "Via rodent contact.",
        "sources": [
            {"title": "Lassa paper", "authors_short": "John et al.", "year": 2024,
             "journal": "J. R. Soc. Interface", "doi": "10.1098/rsif.2024.0106"},
            {"title": "unmapped-file", "authors_short": None, "year": None,
             "journal": None, "doi": None},
        ],
    }
    assert r.headers.get("x-request-id")


@pytest.mark.parametrize(
    "body",
    [{}, {"question": None}, {"question": 123}, {"question": ""},
     {"question": "   "}, {"question": "  hi  "}, {"question": "x" * 501}],
)
def test_invalid_questions_rejected_with_422(body):
    with TestClient(make_app(ok_llm())) as client:
        r = client.post("/ask", json=body)
    assert r.status_code == 422


@pytest.mark.parametrize("question", ["why", "x" * 500, "   why   "])
def test_questions_at_length_limits_accepted(question):
    with TestClient(make_app(ok_llm())) as client:
        assert client.post("/ask", json={"question": question}).status_code == 200


def test_fallback_answer_is_returned():
    llm = raising().with_fallbacks([ok_llm("From fallback.")])
    with TestClient(make_app(llm)) as client:
        r = client.post("/ask", json={"question": "How does Lassa spread?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "From fallback."


def test_all_models_failing_returns_503_without_leaking():
    llm = raising().with_fallbacks([raising()])
    with TestClient(make_app(llm)) as client:
        r = client.post("/ask", json={"question": "How does Lassa spread?"})
    assert r.status_code == 503
    assert r.json() == {"error": "unavailable"}
    assert "gsk_" not in r.text and "internal" not in r.text
    assert r.headers.get("x-request-id")


def test_upstream_429_maps_to_rate_limited():
    llm = raising(429).with_fallbacks([raising(429)])
    with TestClient(make_app(llm)) as client:
        r = client.post("/ask", json={"question": "How does Lassa spread?"})
    assert r.status_code == 429
    assert r.json() == {"error": "rate_limited"}


def test_reasoning_only_output_returns_503():
    with TestClient(make_app(ok_llm("<think>never finishes"))) as client:
        r = client.post("/ask", json={"question": "How does Lassa spread?"})
    assert r.status_code == 503
    assert r.json() == {"error": "unavailable"}


def test_rate_limit_per_client():
    app = make_app(ok_llm(), Settings(rate_limit_per_min=10))
    with TestClient(app) as client:
        a = {"X-Forwarded-For": "203.0.113.1"}
        codes = [client.post("/ask", json={"question": "why?"}, headers=a).status_code
                 for _ in range(11)]
        other = client.post("/ask", json={"question": "why?"},
                            headers={"X-Forwarded-For": "203.0.113.2"})
    assert codes[:10] == [200] * 10
    assert codes[10] == 429
    assert other.status_code == 200


def test_forged_forwarded_for_cannot_dodge_limit_behind_cloudflare():
    app = make_app(ok_llm(), Settings(rate_limit_per_min=2))
    with TestClient(app) as client:
        codes = [
            client.post(
                "/ask", json={"question": "why?"},
                headers={"CF-Connecting-IP": "198.51.100.7",
                         "X-Forwarded-For": f"10.0.0.{i}"},
            ).status_code
            for i in range(3)
        ]
    assert codes == [200, 200, 429]


def test_health_is_503_before_startup_and_200_after():
    app = make_app(ok_llm())
    cold = TestClient(app)  # no context manager -> lifespan has not run
    assert cold.get("/health").status_code == 503
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_cors_allows_only_listed_origins():
    with TestClient(make_app(ok_llm())) as client:
        good = client.options("/ask", headers={
            "Origin": "https://rejusamjohn.pages.dev",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        bad = client.options("/ask", headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        })
    assert good.headers.get("access-control-allow-origin") == "https://rejusamjohn.pages.dev"
    assert "access-control-allow-origin" not in bad.headers


def test_validation_error_body_is_fixed_and_leaks_nothing():
    with TestClient(make_app(ok_llm())) as client:
        r = client.post("/ask", content='{"question": "unterminated', headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json() == {"error": "invalid_question"}
    assert r.headers.get("x-request-id")


def test_short_question_422_body_is_fixed():
    with TestClient(make_app(ok_llm())) as client:
        r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 422
    assert r.json() == {"error": "invalid_question"}


def test_middleware_catches_unhandled_errors_without_leaking():
    app = make_app(ok_llm())

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret")

    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/boom")
    assert r.status_code == 500
    assert r.json() == {"error": "unavailable"}
    assert "secret" not in r.text
    assert r.headers.get("x-request-id")
