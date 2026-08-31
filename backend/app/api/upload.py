"""
Document upload and processing endpoints.

Upload and processing are deliberately separate steps (POST
/documents/upload, then POST /documents/process) rather than one
combined endpoint: uploading is fast and cheap (just saving bytes to
disk), while processing runs the whole embedding pipeline and can take
a few seconds for a large document. Splitting them lets the frontend
show "uploaded" immediately, then a distinct "processing..." state,
rather than one long spinner covering two very different operations.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.config import settings
from app.models.schemas import ProcessRequest, ProcessResponse, UploadResponse
from app.services.chunker import chunk_document
from app.services.document_loader import (
    CorruptedDocumentError,
    DocumentLoader,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.services.document_registry import get_registry
from app.services.embeddings import get_embedding_service
from app.services.vector_store import VectorStore
from app.utils.helpers import generate_document_id, is_valid_document_id, sanitize_filename

router = APIRouter()

# One loader instance is enough -- DocumentLoader holds no per-request
# state, so there's no reason to construct a new one on every call.
_loader = DocumentLoader()


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """Save an uploaded file to disk and register it. Does NOT process
    it yet -- call POST /documents/process next to actually chunk and
    embed it."""
    original_name = file.filename or "unnamed_file"
    extension = Path(original_name).suffix.lower()

    if extension not in _loader.supported_extensions():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{extension}'. "
                f"Supported types: {_loader.supported_extensions()}"
            ),
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024

    # Fast rejection when the client honestly reports Content-Length:
    # refuse before spending any time reading the body at all. This is
    # a best-effort check, not the only one -- a client can lie about
    # or omit this header, which is exactly why the post-read size
    # check below still exists as the real enforcement.
    declared_length = request.headers.get("content-length")
    if declared_length is not None and declared_length.isdigit() and int(declared_length) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_file_size_mb}MB size limit.",
        )

    contents = await file.read()
    size_bytes = len(contents)

    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_file_size_mb}MB size limit.",
        )
    if size_bytes == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    document_id = generate_document_id()
    safe_name = sanitize_filename(original_name)
    saved_path = settings.upload_dir / f"{document_id}_{safe_name}"
    saved_path.write_bytes(contents)

    registry = get_registry()
    registry.add(
        document_id,
        {
            "filename": safe_name,
            "file_type": extension.lstrip("."),
            "size_bytes": size_bytes,
            "status": "uploaded",
            "stored_path": str(saved_path),
        },
    )

    return UploadResponse(
        document_id=document_id,
        filename=safe_name,
        file_type=extension.lstrip("."),
        size_bytes=size_bytes,
        status="uploaded",
    )


@router.post("/documents/process", response_model=ProcessResponse, tags=["documents"])
def process_document(request: ProcessRequest) -> ProcessResponse:
    """Run the full pipeline on a previously uploaded document: extract
    text, chunk it, embed the chunks, build a FAISS index, and save it
    to disk. This is the expensive step -- expect it to take a few
    seconds depending on document size."""
    if not is_valid_document_id(request.document_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document_id format.")

    registry = get_registry()
    record = registry.get(request.document_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No document found with id '{request.document_id}'",
        )

    chunk_size = request.chunk_size or settings.default_chunk_size
    chunk_overlap = request.chunk_overlap or settings.default_chunk_overlap
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    start_time = time.perf_counter()

    try:
        document = _loader.load(record["stored_path"])
    except (UnsupportedFileTypeError, EmptyDocumentError, CorruptedDocumentError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        chunks = chunk_document(document, chunk_size=chunk_size, overlap=chunk_overlap)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document produced no usable chunks.")

    embedder = get_embedding_service()
    vectors = embedder.embed_documents(chunks)

    store = VectorStore(dimension=embedder.embedding_dim)
    store.build(chunks, vectors, embedding_model=embedder.model_name)

    index_dir = settings.index_dir / request.document_id
    store.save(index_dir)

    processing_time = time.perf_counter() - start_time

    updated = registry.update(
        request.document_id,
        status="processed",
        page_count=document.page_count,
        char_count=document.char_count,
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedder.model_name,
    )

    return ProcessResponse(
        document_id=request.document_id,
        filename=updated["filename"],
        page_count=document.page_count,
        char_count=document.char_count,
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedder.model_name,
        processing_time_seconds=round(processing_time, 3),
        status="processed",
    )
