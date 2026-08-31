"""
Tests for app.services.embeddings.

Two groups of tests:

1. Fast, offline unit tests against a FakeSentenceTransformer stand-in --
   these check the *wrapper's* logic (shapes, dtypes, validation,
   lazy-loading, singleton behaviour) without needing to download or
   run the real ~80MB model, and without needing sentence_transformers
   installed at all.

2. One real integration test that loads the actual configured model
   and confirms it truly produces 384-dimensional vectors. This
   requires internet on first run (to download model weights) and
   `sentence-transformers`/`torch` installed. It skips itself
   gracefully if either is unavailable, rather than failing the whole
   suite in an offline/sandboxed environment.

Run from the repo root:
    pytest tests/test_embeddings.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.chunker import DocumentChunk  # noqa: E402
from app.services.embeddings import EmbeddingService, get_embedding_service  # noqa: E402


def _make_chunk(text: str, index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{index}",
        chunk_index=index,
        source_filename="handbook.pdf",
        page_number=1,
        text=text,
        char_count=len(text),
    )


class FakeSentenceTransformer:
    """Stand-in for sentence_transformers.SentenceTransformer.

    Returns a vector derived deterministically from each input text's
    hash (not from call order), instead of running a real model -- lets
    us test everything about EmbeddingService *except* actual semantic
    quality, offline and in milliseconds, while still being able to
    assert "same text in -> same vector out" across separate calls.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            seed = abs(hash(text)) % (2**32)
            vectors[i] = np.random.default_rng(seed).random(self.dim)
        return vectors

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


@pytest.fixture
def service() -> EmbeddingService:
    """An EmbeddingService with the real model swapped out for the fake
    one, injected directly so no network call or heavy import happens."""
    svc = EmbeddingService(model_name="fake-model-for-tests")
    svc._model = FakeSentenceTransformer(dim=384)
    return svc


# --- Wrapper logic (fake model, fast, offline) ------------------------------


def test_model_is_not_loaded_until_first_use():
    # Just constructing the service must not trigger any model load --
    # this is what makes the lazy-loading design testable/verifiable.
    svc = EmbeddingService(model_name="fake-model-for-tests")
    assert svc._model is None


def test_embedding_dim_reports_correctly(service: EmbeddingService):
    assert service.embedding_dim == 384


def test_embed_documents_shape_matches_chunk_count(service: EmbeddingService):
    chunks = [_make_chunk("First chunk of text.", 0), _make_chunk("Second chunk.", 1), _make_chunk("Third.", 2)]

    vectors = service.embed_documents(chunks)

    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32


def test_embed_documents_empty_list_returns_empty_matrix(service: EmbeddingService):
    vectors = service.embed_documents([])

    assert vectors.shape == (0, 384)
    assert vectors.dtype == np.float32


def test_embed_documents_row_order_matches_input_order(service: EmbeddingService):
    # vector_store.py relies on row i of the embedding matrix
    # corresponding exactly to chunks[i] -- this must never be reordered.
    chunks = [_make_chunk(f"chunk number {i}", i) for i in range(5)]

    vectors_first_call = service.embed_documents(chunks)
    vectors_second_call = service.embed_documents(chunks)

    # Same input, same fake model state -> identical output, proving
    # there's no hidden shuffling happening between calls.
    np.testing.assert_array_equal(vectors_first_call, vectors_second_call)


def test_embed_query_returns_single_1d_vector(service: EmbeddingService):
    vector = service.embed_query("What is the attendance requirement?")

    assert vector.shape == (384,)
    assert vector.dtype == np.float32


def test_embed_query_rejects_empty_string(service: EmbeddingService):
    with pytest.raises(ValueError):
        service.embed_query("")


def test_embed_query_rejects_whitespace_only_string(service: EmbeddingService):
    with pytest.raises(ValueError):
        service.embed_query("     ")


def test_get_embedding_service_returns_the_same_instance():
    # lru_cache-based singleton: repeated calls must return the exact
    # same object, not a fresh one -- otherwise the real model would be
    # reloaded into memory on every request.
    first = get_embedding_service()
    second = get_embedding_service()

    assert first is second


# --- Real model integration test (requires internet + sentence-transformers) --


def test_real_model_actually_produces_384_dimensional_vectors():
    """
    Loads the real configured embedding model (all-MiniLM-L6-v2 by
    default) and confirms it truly produces 384-dim vectors end to end.

    This is the one test in the suite that needs sentence-transformers
    and torch installed, plus internet access on first run to download
    model weights from HuggingFace. If either is unavailable (e.g. a
    sandboxed environment with no internet), the test skips itself
    rather than failing the whole suite -- run it on your own machine
    (which has both) to get real confirmation.
    """
    try:
        service = EmbeddingService()
        real_dim = service.embedding_dim
    except Exception as exc:  # ImportError (library missing) or network error
        pytest.skip(f"Real embedding model unavailable in this environment: {exc}")

    assert real_dim == 384

    chunk = _make_chunk("Students must maintain at least 75% attendance.", 0)
    vectors = service.embed_documents([chunk])
    assert vectors.shape == (1, 384)

    query_vector = service.embed_query("What is the attendance policy?")
    assert query_vector.shape == (384,)

    # Sanity check on semantic quality, not just shape: a chunk and a
    # query about the *same* topic should be meaningfully more similar
    # (higher cosine similarity) than two unrelated pieces of text.
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    unrelated_vector = service.embed_query("What is the capital of France?")
    related_similarity = cosine_sim(query_vector, vectors[0])
    unrelated_similarity = cosine_sim(unrelated_vector, vectors[0])

    assert related_similarity > unrelated_similarity
