"""
FastAPI application entry point.

Wires together the routers built across this phase, sets up CORS, and
adds a couple of exception handlers as a safety net (the endpoints
already catch and convert most of these locally into clean
HTTPExceptions -- these handlers exist for any case that slips through
uncaught, so the API never returns an unexplained raw 500 traceback).

Interactive API documentation is generated automatically by FastAPI
at /docs (Swagger UI) and /redoc -- no extra work required.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import documents, health, query, upload
from app.config import settings
from app.services.document_loader import (
    CorruptedDocumentError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.services.generator import GenerationError

app = FastAPI(
    title="RAG Knowledge Assistant API",
    description=(
        "A from-scratch Retrieval-Augmented Generation pipeline: upload a "
        "document, ask questions about it, and get grounded answers with "
        "page-level citations. No LangChain/LlamaIndex -- every stage "
        "(chunking, embedding, retrieval) is hand-written and inspectable."
    ),
    version="0.1.0",
)

# The CORS spec disallows combining a wildcard origin with credentials
# -- browsers will silently reject credentialed requests if both are
# set, which is confusing to debug. Rather than set both and let that
# surprise someone, allow_credentials only turns on once real origins
# are configured (i.e. CORS_ORIGINS is no longer "*").
_origins = settings.cors_origin_list()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(documents.router)


@app.exception_handler(UnsupportedFileTypeError)
@app.exception_handler(EmptyDocumentError)
@app.exception_handler(CorruptedDocumentError)
async def document_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safety net: endpoints already catch these locally, but any spot
    that misses one still gets a clean 400 instead of a raw 500."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


# --- Frontend -------------------------------------------------------------
# Anchored to the project root explicitly (not a bare relative path) --
# this file has already been bitten twice by relative-path bugs that
# only showed up depending on which directory the process was launched
# from, so every filesystem path in this project is now anchored the
# same way, on principle.
#
# Mounted LAST and at "/" so it acts as a catch-all: FastAPI tries
# every explicit route above (/, would-be, /health, /documents, /query,
# /docs, /redoc) first, in the order they were registered, and only
# falls through to serving a static file if nothing else matched.
# html=True makes StaticFiles serve frontend/index.html automatically
# for "/" itself, not just for named files like /style.css or /app.js.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"

if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
