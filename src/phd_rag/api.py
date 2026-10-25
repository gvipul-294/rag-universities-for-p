"""HTTP API for the PhD Finder web frontend."""

from __future__ import annotations

import hmac
import json
from typing import Annotated

import openai
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ingest import CHUNKS_FILE, INDEX_FILE
from .models import CandidateProfile
from .retrieve import RetrievedChunk, answer, profile_to_question
from .settings import Settings, get_settings


class AskRequest(BaseModel):
    """A free-form research question submitted by the Ask tab."""

    question: str = Field(min_length=3, max_length=2_000)
    countries: list[str] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=12, ge=1, le=30)


class MatchRequest(CandidateProfile):
    """The candidate profile submitted by the Match Me tab."""

    top_k: int = Field(default=12, ge=1, le=30)


class SourceResult(BaseModel):
    university: str
    country: str
    url: str
    score: float
    excerpt: str
    fetched_at: str


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    source_count: int
    chat_model: str


def _serialize_sources(results: list[RetrievedChunk]) -> list[SourceResult]:
    return [
        SourceResult(
            university=result.record["university"],
            country=result.record["country"],
            url=result.record["url"],
            score=round(result.score, 4),
            excerpt=result.record["text"],
            fetched_at=result.record["fetched_at"],
        )
        for result in results
    ]


def _index_status(settings: Settings) -> dict[str, object]:
    manifest_path = settings.artifacts_dir / "manifest.json"
    index_path = settings.artifacts_dir / INDEX_FILE
    chunks_path = settings.artifacts_dir / CHUNKS_FILE
    manifest: dict[str, object] | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"warning": "The index manifest is not valid JSON."}
    return {
        "ready": index_path.exists() and chunks_path.exists(),
        "manifest": manifest,
    }


# Server-side setup problems the operator must fix; safe to show (no secrets in these messages).
_SETUP_ERRORS = ("API_KEY is missing", "Index not found", "Index was built with")


def _api_error(error: Exception) -> HTTPException:
    message = str(error)
    if isinstance(error, openai.RateLimitError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The answer model is rate-limited or out of quota. Retry in a minute.",
        )
    if isinstance(error, openai.APIError):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The answer model provider returned an error. Check server logs and retry.",
        )
    if any(marker in message for marker in _SETUP_ERRORS):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=message)
    if isinstance(error, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="The RAG provider could not complete this request. Check server logs and retry.",
    )


def _require_api_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Protect browser-facing endpoints when PHD_RAG_API_TOKEN is configured.

    Leave the token unset only for local development. A production frontend must send
    the bearer token from a server-side proxy, never from browser JavaScript.
    """
    configured_token = get_settings().api_bearer_token
    if not configured_token:
        return
    expected = f"Bearer {configured_token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app() -> FastAPI:
    app = FastAPI(
        title="PhD Finder API",
        version="0.1.0",
        description="Source-backed PhD research leads from the local official-university RAG index.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.get("/api/v1/health", tags=["operations"])
    def health() -> dict[str, object]:
        """Report whether the generated FAISS index is ready to serve requests."""
        return {"service": "phd-finder-api", **_index_status(get_settings())}

    @app.get("/api/v1/index/status", tags=["operations"])
    def index_status(_: None = Depends(_require_api_token)) -> dict[str, object]:
        """Return non-secret index metadata for dashboards and deployment checks."""
        return _index_status(get_settings())

    @app.post("/api/v1/ask", response_model=RAGResponse, tags=["research"])
    def ask(request: AskRequest, _: None = Depends(_require_api_token)) -> RAGResponse:
        """Answer a free-form question from official pages and return source passages."""
        try:
            response_text, results = answer(
                get_settings(),
                request.question,
                top_k=request.top_k,
                countries=request.countries or None,
            )
        except (RuntimeError, ValueError, OSError, openai.APIError) as error:
            raise _api_error(error) from error
        sources = _serialize_sources(results)
        return RAGResponse(
            answer=response_text,
            sources=sources,
            source_count=len(sources),
            chat_model=get_settings().llm_model,
        )

    @app.post("/api/v1/match", response_model=RAGResponse, tags=["research"])
    def match(request: MatchRequest, _: None = Depends(_require_api_token)) -> RAGResponse:
        """Match a structured candidate profile to official PhD research leads."""
        profile = CandidateProfile.model_validate(request.model_dump(exclude={"top_k"}))
        try:
            response_text, results = answer(
                get_settings(),
                profile_to_question(profile),
                top_k=request.top_k,
                countries=profile.preferred_countries or None,
            )
        except (RuntimeError, ValueError, OSError, openai.APIError) as error:
            raise _api_error(error) from error
        sources = _serialize_sources(results)
        return RAGResponse(
            answer=response_text,
            sources=sources,
            source_count=len(sources),
            chat_model=get_settings().llm_model,
        )

    return app


app = create_app()
