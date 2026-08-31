"""
Full API integration tests.

Exercises the actual HTTP endpoints end to end (upload -> process ->
query -> list -> delete) using FastAPI's TestClient. The real
embedding model and real Groq client are swapped out for deterministic
fakes (same pattern as test_embeddings.py / test_retrieval.py /
test_generator.py), so this whole suite runs fast and fully offline --
it's testing OUR wiring (do the endpoints call the right services in
the right order, with the right error handling?), not the ML model's
quality or Groq's API, neither of which is our code to test.

Run from the repo root:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app.api.query as query_module  # noqa: E402
import app.api.upload as upload_module  # noqa: E402
import app.services.document_registry as registry_module  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.embeddings import EmbeddingService  # noqa: E402
from app.services.generator import Generator  # noqa: E402


class _FakeSentenceTransformer:
    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            vectors[i] = np.random.default_rng(abs(hash(text)) % (2**32)).random(self.dim)
        return vectors

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


class _FakeGroqMessage:
    content = "This is a fake grounded answer."


class _FakeGroqChoice:
    message = _FakeGroqMessage()


class _FakeGroqResponse:
    choices = [_FakeGroqChoice()]


class _FakeGroqCompletions:
    """Always returns the same canned answer -- these tests check that
    the API wires retrieval and generation together correctly, not
    what a real LLM would say."""

    def create(self, **kwargs):
        return _FakeGroqResponse()


class _FakeGroqChat:
    completions = _FakeGroqCompletions()


class _FakeGroqClient:
    chat = _FakeGroqChat()


@pytest.fixture(autouse=True)
def isolate_storage(tmp_path, monkeypatch):
    """Redirect all file storage (uploads, indexes, registry) to a
    fresh temp directory for every test, so tests never touch the
    real data/ folder and never leak state between each other."""
    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "indexes"
    upload_dir.mkdir()
    index_dir.mkdir()

    monkeypatch.setattr(settings, "upload_dir", upload_dir)
    monkeypatch.setattr(settings, "index_dir", index_dir)

    registry_module.get_registry.cache_clear()
    yield
    registry_module.get_registry.cache_clear()


@pytest.fixture(autouse=True)
def fake_embedding_and_llm(monkeypatch):
    """Swap the real embedding model and Groq client for fast, offline
    fakes across every test in this file."""
    fake_embedder = EmbeddingService(model_name="fake-for-api-tests")
    fake_embedder._model = _FakeSentenceTransformer(dim=16)

    monkeypatch.setattr(upload_module, "get_embedding_service", lambda: fake_embedder)
    monkeypatch.setattr(query_module, "get_embedding_service", lambda: fake_embedder)

    fake_generator = Generator(api_key="fake-key", model="fake-model")
    fake_generator._client = _FakeGroqClient()
    monkeypatch.setattr(query_module, "get_generator", lambda: fake_generator)

    query_module._vector_store_cache.clear()
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# --- Upload -----------------------------------------------------------


def test_upload_accepts_valid_txt_file(client: TestClient):
    response = client.post(
        "/documents/upload",
        files={"file": ("policy.txt", b"Attendance must be 75 percent.", "text/plain")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "policy.txt"
    assert data["status"] == "uploaded"
    assert "document_id" in data


def test_upload_rejects_unsupported_file_type(client: TestClient):
    response = client.post(
        "/documents/upload",
        files={"file": ("data.csv", b"a,b,c", "text/csv")},
    )
    assert response.status_code == 400


def test_upload_rejects_empty_file(client: TestClient):
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_sanitizes_malicious_filename(client: TestClient):
    response = client.post(
        "/documents/upload",
        files={"file": ("../../etc/passwd.txt", b"some content", "text/plain")},
    )

    assert response.status_code == 201
    # Path components must be stripped -- the stored filename should
    # never contain a directory traversal sequence.
    assert ".." not in response.json()["filename"]
    assert "/" not in response.json()["filename"]


# --- Process ------------------------------------------------------------


def test_process_unknown_document_returns_404(client: TestClient):
    # A well-formed but nonexistent id (12 lowercase hex chars) --
    # this tests the "not found" path specifically, distinct from the
    # "malformed id" path covered in test_security.py.
    response = client.post("/documents/process", json={"document_id": "000000000000"})
    assert response.status_code == 404


def test_process_rejects_overlap_not_smaller_than_chunk_size(client: TestClient):
    upload_resp = client.post(
        "/documents/upload",
        files={"file": ("a.txt", b"Some content here to process.", "text/plain")},
    )
    document_id = upload_resp.json()["document_id"]

    response = client.post(
        "/documents/process",
        json={"document_id": document_id, "chunk_size": 100, "chunk_overlap": 200},
    )
    assert response.status_code == 400


def test_process_succeeds_and_returns_statistics(client: TestClient):
    content = b"Attendance Policy\n\nStudents must maintain 75 percent attendance to sit the exam."
    upload_resp = client.post("/documents/upload", files={"file": ("handbook.txt", content, "text/plain")})
    document_id = upload_resp.json()["document_id"]

    response = client.post(
        "/documents/process",
        json={"document_id": document_id, "chunk_size": 200, "chunk_overlap": 20},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processed"
    assert data["chunk_count"] >= 1
    assert data["char_count"] > 0
    assert data["processing_time_seconds"] >= 0


# --- Query ---------------------------------------------------------------


def test_query_unknown_document_returns_404(client: TestClient):
    response = client.post("/query", json={"document_id": "000000000000", "question": "anything?"})
    assert response.status_code == 404


def test_query_unprocessed_document_returns_400(client: TestClient):
    upload_resp = client.post("/documents/upload", files={"file": ("a.txt", b"Some content.", "text/plain")})
    document_id = upload_resp.json()["document_id"]

    response = client.post("/query", json={"document_id": document_id, "question": "anything?"})
    assert response.status_code == 400


def test_query_rejects_empty_question(client: TestClient):
    upload_resp = client.post("/documents/upload", files={"file": ("a.txt", b"Some content here.", "text/plain")})
    document_id = upload_resp.json()["document_id"]
    client.post("/documents/process", json={"document_id": document_id})

    response = client.post("/query", json={"document_id": document_id, "question": ""})
    # Pydantic's min_length=1 rejects this before it even reaches our code.
    assert response.status_code == 422


def test_full_upload_process_query_round_trip(client: TestClient):
    content = b"Attendance Policy\n\nStudents must maintain 75 percent attendance to sit the exam."
    upload_resp = client.post("/documents/upload", files={"file": ("handbook.txt", content, "text/plain")})
    document_id = upload_resp.json()["document_id"]

    process_resp = client.post(
        "/documents/process",
        json={"document_id": document_id, "chunk_size": 200, "chunk_overlap": 20},
    )
    assert process_resp.status_code == 200

    query_resp = client.post(
        "/query",
        json={"document_id": document_id, "question": "What is the attendance policy?", "top_k": 2},
    )

    assert query_resp.status_code == 200
    data = query_resp.json()
    assert data["answer"] == "This is a fake grounded answer."
    assert len(data["sources"]) >= 1
    assert "page_number" in data["sources"][0]
    assert "similarity" in data["sources"][0]


# --- List / delete --------------------------------------------------------


def test_list_documents_includes_uploaded_document(client: TestClient):
    upload_resp = client.post("/documents/upload", files={"file": ("a.txt", b"some content", "text/plain")})
    document_id = upload_resp.json()["document_id"]

    response = client.get("/documents")

    assert response.status_code == 200
    ids = [d["document_id"] for d in response.json()["documents"]]
    assert document_id in ids


def test_delete_document_removes_it_from_the_list(client: TestClient):
    upload_resp = client.post("/documents/upload", files={"file": ("a.txt", b"some content", "text/plain")})
    document_id = upload_resp.json()["document_id"]

    delete_resp = client.delete(f"/documents/{document_id}")
    assert delete_resp.status_code == 204

    list_resp = client.get("/documents")
    ids = [d["document_id"] for d in list_resp.json()["documents"]]
    assert document_id not in ids


def test_delete_unknown_document_returns_404(client: TestClient):
    response = client.delete("/documents/000000000000")
    assert response.status_code == 404


def test_deleted_document_cannot_be_queried(client: TestClient):
    content = b"Some content to process and then delete."
    upload_resp = client.post("/documents/upload", files={"file": ("a.txt", content, "text/plain")})
    document_id = upload_resp.json()["document_id"]
    client.post("/documents/process", json={"document_id": document_id})

    client.delete(f"/documents/{document_id}")

    response = client.post("/query", json={"document_id": document_id, "question": "anything?"})
    assert response.status_code == 404
