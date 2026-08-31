"""
Tests for the /health and / endpoints.

Run from the repo root:
    pytest tests/test_health.py -v
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_returns_ok_status():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_reports_configuration_without_exposing_secrets():
    response = client.get("/health")
    data = response.json()

    assert "embedding_model" in data
    assert "vector_store" in data
    assert "llm_provider" in data
    assert isinstance(data["groq_configured"], bool)
    # Must report WHETHER a key is set, never the key value itself.
    assert "groq_api_key" not in data


def test_root_endpoint_serves_the_frontend():
    # As of Phase 7, "/" is served by the frontend's StaticFiles mount
    # (index.html), not a JSON handler -- this test documents that
    # deliberate change rather than asserting the old JSON shape.
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_interactive_docs_are_available():
    response = client.get("/docs")
    assert response.status_code == 200
