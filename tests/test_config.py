from __future__ import annotations

import os

import pytest

from rag_api.config import Settings, load_env_file


def test_defaults_match_spec():
    s = Settings.from_env({})
    assert s.groq_model == "openai/gpt-oss-120b"
    assert s.groq_fallback_model == "openai/gpt-oss-20b"
    assert s.llm_timeout_s == 20.0
    assert s.rate_limit_per_min == 10
    assert "https://rejusamjohn.pages.dev" in s.allowed_origins
    assert "http://localhost:5500" in s.allowed_origins


def test_env_overrides():
    s = Settings.from_env(
        {
            "GROQ_MODEL": "m1",
            "GROQ_FALLBACK_MODEL": "m2",
            "LLM_TIMEOUT_S": "12.5",
            "RATE_LIMIT_PER_MIN": "3",
            "ALLOWED_ORIGINS": " https://a.example , ,https://b.example ",
        }
    )
    assert (s.groq_model, s.groq_fallback_model) == ("m1", "m2")
    assert s.llm_timeout_s == 12.5
    assert s.rate_limit_per_min == 3
    assert s.allowed_origins == ("https://a.example", "https://b.example")


@pytest.mark.parametrize(
    "env",
    [
        {"RATE_LIMIT_PER_MIN": "abc"},
        {"RATE_LIMIT_PER_MIN": "0"},
        {"LLM_TIMEOUT_S": "-1"},
        {"GROQ_MODEL": "   "},
    ],
)
def test_invalid_values_fail_loudly(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_load_env_file_sets_missing_keys_only(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nRAG_TEST_A=from_file\nRAG_TEST_B='quoted'\n\nRAG_TEST_C=from_file\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RAG_TEST_A", raising=False)
    monkeypatch.delenv("RAG_TEST_B", raising=False)
    monkeypatch.setenv("RAG_TEST_C", "already_set")

    load_env_file(env_file)

    assert os.environ["RAG_TEST_A"] == "from_file"
    assert os.environ["RAG_TEST_B"] == "quoted"
    assert os.environ["RAG_TEST_C"] == "already_set"


def test_load_env_file_missing_is_noop(tmp_path):
    load_env_file(tmp_path / "nope.env")
