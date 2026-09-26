"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ORIGINS: tuple[str, ...] = (
    "https://rejusamjohn.pages.dev",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
)


@dataclass(frozen=True)
class Settings:
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    llm_timeout_s: float = 20.0
    rate_limit_per_min: int = 10
    allowed_origins: tuple[str, ...] = DEFAULT_ORIGINS

    def __post_init__(self) -> None:
        if not self.groq_model.strip() or not self.groq_fallback_model.strip():
            raise ValueError("GROQ_MODEL and GROQ_FALLBACK_MODEL must be non-empty")
        if self.llm_timeout_s <= 0:
            raise ValueError("LLM_TIMEOUT_S must be positive")
        if self.rate_limit_per_min < 1:
            raise ValueError("RATE_LIMIT_PER_MIN must be at least 1")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        origins = env.get("ALLOWED_ORIGINS")
        return cls(
            groq_model=env.get("GROQ_MODEL", cls.groq_model),
            groq_fallback_model=env.get(
                "GROQ_FALLBACK_MODEL", cls.groq_fallback_model
            ),
            llm_timeout_s=float(env.get("LLM_TIMEOUT_S", cls.llm_timeout_s)),
            rate_limit_per_min=int(
                env.get("RATE_LIMIT_PER_MIN", cls.rate_limit_per_min)
            ),
            allowed_origins=(
                tuple(o.strip() for o in origins.split(",") if o.strip())
                if origins
                else DEFAULT_ORIGINS
            ),
        )


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing values.

    Only for local runs (live tests, evaluation). Production reads real
    environment variables set in the Render dashboard.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
