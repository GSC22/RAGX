"""
Pydantic models for every API request and response.

Kept in one file, separate from the endpoints themselves, so the
"shape" of the API is readable in one place without wading through
endpoint logic -- and so FastAPI's auto-generated /docs page has a
single source of truth for these types.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# --- Upload -----------------------------------------------------------


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size_bytes: int
    status: str


# --- Process ------------------------------------------------------------


class ProcessRequest(BaseModel):
    document_id: str
    # Both optional: omitting them falls back to the configured
    # defaults (settings.default_chunk_size / default_chunk_overlap),
    # so a client can process a document without needing to know or
    # care what those defaults are. Upper bounds prevent a request
    # from asking for a pathologically huge chunk size/overlap that
    # would waste memory and time for no real benefit.
    chunk_size: Optional[int] = Field(default=None, gt=0, le=8000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=4000)


class ProcessResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    char_count: int
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    processing_time_seconds: float
    status: str


# --- Query ---------------------------------------------------------------


class QueryRequest(BaseModel):
    document_id: str
    # max_length caps how much a single request can cost in tokens
    # sent to the LLM; top_k's upper bound prevents a request from
    # asking for an unreasonable number of chunks to retrieve and feed
    # into the prompt.
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, gt=0, le=50)


class SourceItem(BaseModel):
    """One cited chunk backing an answer -- always traced directly
    from what the retriever actually found, never fabricated."""

    chunk_id: str
    page_number: int
    similarity: float
    text: str


class QueryResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    sources: List[SourceItem]


# --- Documents (list / delete) -----------------------------------------


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    uploaded_at: str
    page_count: Optional[int] = None
    char_count: Optional[int] = None
    chunk_count: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    embedding_model: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


# --- Health ------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    vector_store: str
    llm_provider: str
    groq_configured: bool
