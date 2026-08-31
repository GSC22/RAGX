"""
Tests for app.services.retriever.

Uses a deterministic fake embedding model (same pattern as
test_embeddings.py) so the whole embed-then-search path can be tested
fast and offline: identical input text always maps to the identical
vector, so querying with the exact text of a known chunk should
retrieve that chunk first with similarity ~1.0.

Run from the repo root:
    pytest tests/test_retrieval.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.chunker import DocumentChunk  # noqa: E402
from app.services.embeddings import EmbeddingService  # noqa: E402
from app.services.retriever import Retriever  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402


class FakeSentenceTransformer:
    """Deterministic stand-in: identical text -> identical vector,
    different text -> (essentially certainly) different vector. See
    test_embeddings.py for the same pattern with more explanation."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            seed = abs(hash(text)) % (2**32)
            vectors[i] = np.random.default_rng(seed).random(self.dim)
        return vectors

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


def _make_chunk(index: int, text: str, page: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{index}",
        chunk_index=index,
        source_filename="handbook.txt",
        page_number=page,
        text=text,
        char_count=len(text),
    )


@pytest.fixture
def retriever() -> Retriever:
    embedder = EmbeddingService(model_name="fake-model-for-tests")
    embedder._model = FakeSentenceTransformer(dim=16)

    chunks = [
        _make_chunk(0, "Attendance must be at least 75 percent", page=1),
        _make_chunk(1, "The library allows borrowing five books", page=2),
        _make_chunk(2, "Grading is forty percent continuous assessment", page=3),
    ]
    embeddings = embedder.embed_documents(chunks)

    store = VectorStore(dimension=embedder.embedding_dim)
    store.build(chunks, embeddings, embedding_model="fake-model-for-tests")

    return Retriever(embedding_service=embedder, vector_store=store)


def test_retriever_finds_exact_text_match_as_top_result(retriever: Retriever):
    results = retriever.retrieve("The library allows borrowing five books", top_k=3)

    assert len(results) == 3
    assert results[0].text == "The library allows borrowing five books"
    assert results[0].page_number == 2
    assert results[0].rank == 1
    assert results[0].similarity == pytest.approx(1.0, abs=1e-4)


def test_retriever_respects_top_k(retriever: Retriever):
    results = retriever.retrieve("attendance requirement", top_k=1)
    assert len(results) == 1


def test_retriever_results_sorted_descending_by_similarity(retriever: Retriever):
    results = retriever.retrieve("something about grading", top_k=3)

    scores = [r.similarity for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_ranks_are_sequential_starting_at_one(retriever: Retriever):
    results = retriever.retrieve("test query", top_k=3)
    assert [r.rank for r in results] == [1, 2, 3]


def test_retriever_rejects_non_positive_top_k(retriever: Retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("anything", top_k=0)


def test_retriever_rejects_empty_query(retriever: Retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("", top_k=3)


def test_retrieved_chunks_carry_correct_source_filename(retriever: Retriever):
    results = retriever.retrieve("library books", top_k=1)
    assert results[0].source_filename == "handbook.txt"
