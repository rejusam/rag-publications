"""Live checks against real Groq. Run manually: pytest -m live"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_api.config import Settings, load_env_file

pytestmark = pytest.mark.live
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module", autouse=True)
def _env():
    load_env_file(ROOT / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from rag_api.app import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.mark.parametrize(
    "question",
    [
        "How does Lassa virus spread from rodents to humans?",
        "Does travel time matter for pandemic spread in highly connected networks?",
        "Can Ebola persist in non-human primate populations?",
    ],
)
def test_live_answer_has_text_and_sources(client, question):
    r = client.post("/ask", json={"question": question})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].strip()
    assert body["sources"] and body["sources"][0]["doi"]
    assert "<think>" not in body["answer"].lower()
    print(f"\nQ: {question}\nA: {body['answer']}\nSources: {body['sources']}")


@pytest.mark.parametrize("model_attr", ["groq_model", "groq_fallback_model"])
def test_each_model_keeps_reasoning_out_of_content(model_attr):
    from langchain_groq import ChatGroq

    model = getattr(Settings.from_env(), model_attr)
    msg = ChatGroq(model=model, temperature=0.1, timeout=30).invoke(
        "What is 17 + 25? Reply with the number only."
    )
    reasoning = str(msg.additional_kwargs.get("reasoning_content", ""))
    assert "42" in msg.content
    assert len(msg.content) < 40, f"content looks like it includes reasoning: {msg.content!r}"
    if reasoning:
        assert reasoning not in msg.content
