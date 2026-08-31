"""
Retriever service.

Turns a user's question into an embedding and searches a VectorStore
for the most relevant chunks. This is intentionally a thin
coordinating layer -- all the real work (embedding, normalized
similarity search) already lives in embeddings.py and vector_store.py.
Keeping this file thin is what makes each piece independently
testable and easy to explain: "the retriever's whole job is two
function calls in the right order."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    """One retrieved chunk plus how it ranked for a specific query.

    This is the shape that flows into the LLM prompt (Phase 5) and
    into the API's "sources" response (Phase 6) -- it carries
    everything needed to cite the chunk honestly (filename, page,
    similarity) without the caller needing to reach back into the
    vector store.
    """

    rank: int  # 1 = most similar
    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    similarity: float


class Retriever:
    """Combines an embedding model and a vector store to answer:
    'which chunks are most relevant to this question?'
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Embed the query and return its top_k most similar chunks,
        ranked most to least similar.

        Raises:
            ValueError: propagated from embed_query (empty query) or
                from the vector store (non-positive top_k).
        """
        query_vector = self.embedding_service.embed_query(query)
        raw_results = self.vector_store.search(query_vector, top_k)

        return [
            RetrievedChunk(
                rank=rank,
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                source_filename=meta["source_filename"],
                page_number=meta["page_number"],
                similarity=score,
            )
            for rank, (score, meta) in enumerate(raw_results, start=1)
        ]
