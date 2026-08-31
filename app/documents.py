"""
List and delete document endpoints.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.api.query import invalidate_vector_store_cache
from app.config import settings
from app.models.schemas import DocumentInfo, DocumentListResponse
from app.services.document_registry import get_registry
from app.utils.helpers import is_valid_document_id

router = APIRouter()


def _to_document_info(record: dict) -> DocumentInfo:
    return DocumentInfo(
        document_id=record["document_id"],
        filename=record["filename"],
        file_type=record["file_type"],
        status=record["status"],
        uploaded_at=record["uploaded_at"],
        page_count=record.get("page_count"),
        char_count=record.get("char_count"),
        chunk_count=record.get("chunk_count"),
        chunk_size=record.get("chunk_size"),
        chunk_overlap=record.get("chunk_overlap"),
        embedding_model=record.get("embedding_model"),
    )


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents() -> DocumentListResponse:
    registry = get_registry()
    records = registry.list_all()
    return DocumentListResponse(documents=[_to_document_info(r) for r in records])


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    tags=["documents"],
)
def delete_document(document_id: str) -> None:
    """Remove a document's uploaded file, its saved FAISS index, and
    its registry entry. Also evicts it from the query endpoint's
    in-memory cache so a deleted document can never be queried
    against a stale cached index."""
    if not is_valid_document_id(document_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document_id format.")

    registry = get_registry()
    record = registry.get(document_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document found with id '{document_id}'",
        )

    stored_path = record.get("stored_path")
    if stored_path:
        Path(stored_path).unlink(missing_ok=True)

    index_dir = settings.index_dir / document_id
    if index_dir.exists():
        shutil.rmtree(index_dir)

    invalidate_vector_store_cache(document_id)
    registry.delete(document_id)
