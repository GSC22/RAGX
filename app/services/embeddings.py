"""
Embedding service.

Wraps a sentence-transformers model that runs entirely locally (no
network calls at inference time, no API key) to turn chunk text and
user queries into vectors that live in the same vector space -- which
is what makes similarity search meaningful in the first place.

Design notes:

- The actual model is loaded lazily, both at the `sentence_transformers`
  import level and the model-weights level (see `model` property).
  `sentence_transformers` pulls in `torch`, a large and slow-to-import
  library; importing it lazily means code that only exercises this
  wrapper's logic (shape handling, validation) never pays that cost,
  and the FastAPI app doesn't stall on startup loading it before it's
  needed.

- Normalization (scaling vectors to unit length so cosine similarity
  can be computed via a simple dot product) is intentionally NOT done
  here. That's a concern of the vector store, not the embedder --
  keeping this class limited to "text in, raw vector out" keeps it
  simple to test and easy to swap models on later.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from app.config import settings
from app.services.chunker import DocumentChunk

if TYPE_CHECKING:
    # Only imported for type checkers; never actually executed at
    # runtime, so this doesn't force sentence_transformers/torch to be
    # installed just to import this module.
    from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Turns text into fixed-size vectors using a local HuggingFace model.

    Why sentence-transformers/all-MiniLM-L6-v2 (the default):
    - 384-dimensional output: small enough to keep a FAISS index for a
      few-hundred-page document comfortably in memory, large enough to
      capture meaningful semantic distinctions.
    - ~80MB download, runs fast on CPU: no GPU required, which matters
      for a project meant to run on a student's laptop or a free-tier
      deployment.
    - Trained specifically for sentence/short-passage similarity, which
      is exactly our chunk size range (a few hundred characters) --
      unlike raw word-embedding models, it produces one embedding per
      passage that reflects overall meaning, not per-token vectors.
    - Trade-off, stated plainly: bigger models (e.g. bge-small-en-v1.5,
      768-dim, or larger) generally retrieve slightly more accurately
      but cost more memory/compute per query. For a single-document,
      student-scale RAG system, that accuracy gain is rarely worth the
      slower iteration loop during development.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: "SentenceTransformer | None" = None

    @property
    def model(self) -> "SentenceTransformer":
        """Load the model on first access, then reuse it.

        Loading (downloading, on first-ever run, then reading weights
        into memory) takes a few seconds -- doing this once per process
        rather than once per request is why `get_embedding_service()`
        below is a singleton.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: see module docstring

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of vectors this model produces (384 for the default model)."""
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Embed a list of chunks into a (N, embedding_dim) float32 matrix.

        Row i of the returned matrix corresponds to chunks[i] -- this
        ordering is relied on by vector_store.py when building the
        FAISS index, so it must never be silently reordered here.
        """
        if not chunks:
            return np.empty((0, self.embedding_dim), dtype="float32")
        texts = [chunk.text for chunk in chunks]
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user question into a (embedding_dim,) float32 vector."""
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")
        return self._encode([query])[0]

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Shared encoding path for both documents and queries.

        batch_size=32 lets sentence-transformers process multiple texts
        per forward pass instead of one at a time, which matters when
        embedding a document that produced, say, 200 chunks.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Process-wide singleton so the model is loaded into memory once,
    not once per request. Used as a FastAPI dependency in later phases."""
    return EmbeddingService()
