"""HTTP layer: validation, rate limiting, error mapping and health."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints

from rag_api.config import Settings
from rag_api.pipeline import Pipeline, build_pipeline
from rag_api.ratelimit import FixedWindowLimiter

logger = logging.getLogger("rag_api")

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)
]


class AskRequest(BaseModel):
    question: Question


class SourceOut(BaseModel):
    title: str
    authors_short: str | None = None
    year: int | None = None
    journal: str | None = None
    doi: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


def client_ip(request: Request) -> str:
    """Best-effort client address.

    Cloudflare sits in front of Render and overwrites CF-Connecting-IP, so it
    can't be forged by the caller. The first X-Forwarded-For entry can be,
    so it's only a fallback.
    """
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse({"error": code}, status_code=status)


def create_app(
    settings: Settings | None = None,
    pipeline_factory: Callable[[Settings], Pipeline] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    factory = pipeline_factory or build_pipeline
    limiter = FixedWindowLimiter(settings.rate_limit_per_min)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = factory(settings)
        logger.info(
            "Pipeline ready (model=%s, fallback=%s)",
            settings.groq_model,
            settings.groq_fallback_model,
        )
        yield

    app = FastAPI(title="Ask My Research API", lifespan=lifespan)
    app.state.pipeline = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "unhandled error (request_id=%s)", request.state.request_id
            )
            response = _error(500, "unavailable")
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "validation error (request_id=%s): %s", request_id, exc.errors()
        )
        return _error(422, "invalid_question")

    @app.get("/health")
    def health():
        if app.state.pipeline is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request):
        if not limiter.allow(client_ip(request)):
            return _error(429, "rate_limited")
        pipeline = app.state.pipeline
        if pipeline is None:
            return _error(503, "unavailable")
        try:
            result = pipeline.ask(body.question)
        except Exception as exc:
            logger.exception("ask failed (request_id=%s)", request.state.request_id)
            if getattr(exc, "status_code", None) == 429:
                return _error(429, "rate_limited")
            return _error(503, "unavailable")
        return AskResponse(
            answer=result.text,
            sources=[SourceOut(**asdict(s)) for s in result.sources],
        )

    return app
