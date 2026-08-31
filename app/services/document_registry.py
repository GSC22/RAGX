"""
Document registry.

A small JSON-backed "database" recording what's been uploaded and
processed, so GET /documents and DELETE /documents/{id} have something
to read and modify, and POST /documents/process can look up where a
previously-uploaded file actually lives on disk.

This is intentionally NOT a real database: no transactions, no
concurrent-write safety beyond a single in-process lock. For a
single-user, single-process student project that's a reasonable
trade-off -- but it's a real limitation, not a hidden one, and it's
called out here plainly (and again in the README) rather than
presented as more robust than it is. Swapping this for SQLite or
Postgres later would only require changing this one file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Union

from app.config import settings


class DocumentRegistry:
    """Reads and writes a single JSON file mapping document_id -> record."""

    def __init__(self, registry_path: Union[str, Path]) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Guards against two requests in the same process corrupting the
        # file via an interleaved read-modify-write. Does NOT protect
        # against multiple separate processes (e.g. multiple uvicorn
        # workers) writing at once -- a known limitation for a
        # single-worker dev deployment.
        self._lock = Lock()
        if not self.registry_path.exists():
            self._write({})

    def _read(self) -> Dict[str, dict]:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict[str, dict]) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, document_id: str, record: dict) -> dict:
        """Register a newly uploaded document."""
        with self._lock:
            data = self._read()
            full_record = {
                **record,
                "document_id": document_id,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            data[document_id] = full_record
            self._write(data)
            return full_record

    def update(self, document_id: str, **fields) -> dict:
        """Merge new fields into an existing record (e.g. after processing)."""
        with self._lock:
            data = self._read()
            if document_id not in data:
                raise KeyError(f"No document registered with id '{document_id}'")
            data[document_id].update(fields)
            self._write(data)
            return data[document_id]

    def get(self, document_id: str) -> Optional[dict]:
        with self._lock:
            data = self._read()
            return data.get(document_id)

    def list_all(self) -> List[dict]:
        with self._lock:
            data = self._read()
            return list(data.values())

    def delete(self, document_id: str) -> None:
        with self._lock:
            data = self._read()
            data.pop(document_id, None)
            self._write(data)


@lru_cache(maxsize=1)
def get_registry() -> DocumentRegistry:
    """Process-wide singleton, used as a FastAPI dependency.

    The registry file lives alongside uploads/ and indexes/ under the
    project's data/ directory, not inside either of them.
    """
    return DocumentRegistry(settings.upload_dir.parent / "documents_registry.json")
