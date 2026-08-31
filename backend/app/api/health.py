"""
Health check endpoint.

Useful both as a deployment liveness check and as a quick way to
confirm the app's configuration without exposing secrets: it reports
WHETHER a Groq key is configured (True/False), never the key itself.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        embedding_model=settings.embedding_model,
        vector_store="faiss (IndexFlatIP, cosine via normalized inner product)",
        llm_provider="groq",
        groq_configured=bool(settings.groq_api_key),
    )
