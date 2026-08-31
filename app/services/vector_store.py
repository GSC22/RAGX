"""
Vector store service.

Wraps a FAISS IndexFlatIP (exact, brute-force inner-product search) to
store and search document chunk embeddings, plus a parallel JSON file
holding each chunk's metadata (text, filename, page number). FAISS
itself only stores raw vectors and integer ids -- it knows nothing
about "page 12" or "handbook.pdf" -- so that information has to live
somewhere else, kept in sync by us.

Why IndexFlatIP + normalization for cosine similarity:
FAISS has no dedicated "cosine similarity" index type. The standard
workaround: L2-normalize every vector to unit length before indexing
and before searching. Once every vector has length 1, the inner
product between two vectors is mathematically identical to their
cosine similarity (cos(theta) = (a . b) / (|a| |b|), and |a| = |b| = 1
after normalization, so cos(theta) = a . b exactly). IndexFlatIP
computes that inner product, brute-force, against every stored vector
-- which is *exact* (not an approximation like IVF/HNSW indexes use),
and is plenty fast for the document sizes this project targets (tens
to a few hundred chunks per document).

Why metadata is stored separately (metadata.json) rather than "in" FAISS:
FAISS assigns each added vector a sequential integer id (0, 1, 2, ...)
in insertion order and stores nothing else. So a plain Python list is
kept where metadata[i] describes exactly the vector FAISS knows as id
i. This positional 1:1 mapping is the entire trick, and it's why
chunks must always be embedded and added in a fixed, unchanged order
(embeddings.py's docstring makes the same promise from the other side).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Union

import faiss
import numpy as np

from app.services.chunker import DocumentChunk


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit length (see module docstring for why).

    Guards against dividing by zero for a pathological all-zero vector
    (which would otherwise produce NaNs and silently corrupt the index).
    """
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return (vectors / norms).astype("float32")


class VectorStore:
    """A FAISS IndexFlatIP index plus its aligned chunk metadata."""

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "metadata.json"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[dict] = []
        self._embedding_model = ""

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self.index.ntotal

    def build(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
        embedding_model: str = "",
    ) -> None:
        """Build the index from a document's chunks and their embeddings.

        Args:
            chunks: chunks in the SAME order as the embeddings matrix rows
                -- this ordering is what makes vector id i correspond to
                chunks[i], and it is not re-verified beyond the count
                check below, so callers must not reorder one without
                the other.
            embeddings: (N, dimension) float32 matrix from EmbeddingService.
            embedding_model: recorded in the manifest, purely for later
                reference/debugging (e.g. "which model produced this index?").

        Raises:
            ValueError: if the chunk count doesn't match the embedding
                row count, or the embedding dimension doesn't match
                this store's configured dimension.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding row "
                f"count ({embeddings.shape[0]}) -- they must be in the same order."
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match "
                f"this store's configured dimension ({self.dimension})."
            )

        normalized = normalize_vectors(embeddings)
        self.index.add(normalized)
        self.metadata = [asdict(chunk) for chunk in chunks]
        self._embedding_model = embedding_model

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[float, dict]]:
        """Search for the top_k most similar chunks to a query vector.

        Returns a list of (similarity_score, metadata_dict) tuples,
        ordered from most to least similar. similarity_score is a
        cosine similarity, mathematically in [-1, 1] but in practice
        close to [0, 1] for real sentence embeddings of related text.

        top_k is silently clamped to the number of stored vectors if
        it asks for more results than exist -- this is a normal,
        expected situation (e.g. a 3-chunk document with top_k=5), not
        an error.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.size == 0:
            return []

        effective_k = min(top_k, self.size)
        normalized_query = normalize_vectors(query_vector)
        scores, ids = self.index.search(normalized_query, effective_k)

        results: List[Tuple[float, dict]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:  # FAISS pads with -1 if fewer than top_k results exist
                continue
            results.append((float(score), self.metadata[idx]))
        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persist the index, metadata, and a small manifest to disk.

        Three files are written so the store can be fully reconstructed
        later by `load()`:
          - index.faiss   : the actual FAISS index (vectors)
          - metadata.json : list of chunk metadata dicts, position-aligned
                             with the FAISS vector ids
          - manifest.json : bookkeeping (dimension, count, which
                             embedding model produced these vectors, when)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / self.INDEX_FILENAME))

        with open(directory / self.METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        manifest = {
            "dimension": self.dimension,
            "num_vectors": self.size,
            "embedding_model": self._embedding_model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(directory / self.MANIFEST_FILENAME, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "VectorStore":
        """Load a previously saved index + metadata from disk.

        Raises:
            FileNotFoundError: if no saved index exists at this path
                (e.g. a bad or stale document_id) -- callers should
                turn this into a clear 404 at the API layer, not let
                it surface as an unexplained 500.
        """
        directory = Path(directory)
        index_path = directory / cls.INDEX_FILENAME
        metadata_path = directory / cls.METADATA_FILENAME

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"No saved vector store found at {directory}")

        index = faiss.read_index(str(index_path))
        store = cls(dimension=index.d)
        store.index = index

        with open(metadata_path, "r", encoding="utf-8") as f:
            store.metadata = json.load(f)

        manifest_path = directory / cls.MANIFEST_FILENAME
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            store._embedding_model = manifest.get("embedding_model", "")

        return store
