"""
Query endpoint: the payoff of the whole pipeline.

Loads a document's saved FAISS index, retrieves the top-k most
relevant chunks for the question, and asks Groq to generate a grounded
answer from them.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.models.schemas import QueryRequest, QueryResponse, SourceItem
from app.services.document_registry import get_registry
from app.services.embeddings import get_embedding_service
from app.services.generator import GenerationError, get_generator
from app.services.retriever import Retriever
from app.services.vector_store import VectorStore
from app.utils.helpers import is_valid_document_id

router = APIRouter()

# Small in-memory cache so repeated questions against the same document
# don't re-read index.faiss + metadata.json from disk every single
# time. Keyed by document_id; api/documents.py's delete endpoint
# invalidates an entry here when that document is removed, so a
# deleted document can never be queried against a stale cached index.
_vector_store_cache: dict[str, VectorStore] = {}


def _load_vector_store(document_id: str) -> VectorStore:
    if document_id not in _vector_store_cache:
        index_dir = settings.index_dir / document_id
        try:
            _vector_store_cache[document_id] = VectorStore.load(index_dir)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No processed index found for document '{document_id}'. "
                    "Call /documents/process first."
                ),
            ) from exc
    return _vector_store_cache[document_id]


def invalidate_vector_store_cache(document_id: str) -> None:
    """Called by api/documents.py when a document is deleted."""
    _vector_store_cache.pop(document_id, None)


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query_document(request: QueryRequest) -> QueryResponse:
    if not is_valid_document_id(request.document_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document_id format.")

    registry = get_registry()
    record = registry.get(request.document_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document found with id '{request.document_id}'",
        )
    if record.get("status") != "processed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has not been processed yet. Call /documents/process first.",
        )

    top_k = request.top_k or settings.default_top_k

    store = _load_vector_store(request.document_id)
    embedder = get_embedding_service()
    retriever = Retriever(embedding_service=embedder, vector_store=store)

    try:
        results = retriever.retrieve(request.question, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    generator = get_generator()
    try:
        answer = generator.generate_answer(request.question, results)
    except GenerationError as exc:
        # 502: our server is fine, but the upstream (Groq) call failed
        # -- a missing/invalid API key or a network issue on our end,
        # not something wrong with the caller's request.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return QueryResponse(
        document_id=request.document_id,
        question=request.question,
        answer=answer,
        sources=[
            SourceItem(
                chunk_id=r.chunk_id,
                page_number=r.page_number,
                similarity=round(r.similarity, 4),
                text=r.text,
            )
            for r in results
        ],
    )
