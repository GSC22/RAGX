"""
Document ingestion service.

Responsible ONLY for turning a file on disk into plain text plus
page-level metadata. It knows nothing about chunking, embeddings, or
retrieval -- keeping this boundary clean is what lets us add a new
file type (e.g. .docx, .html) later by registering one function,
without touching any other part of the pipeline.

Design notes for the interview:
- `LoadedDocument.pages` always has at least one page, even for
  plain-text files -- a .txt/.md file is treated as a single logical
  "page" (page_number=1) so downstream code (the chunker, the source
  citations) can use one consistent data shape regardless of file type.
- Extraction failures raise specific exceptions instead of returning
  None/empty results, so the API layer can turn them into meaningful
  HTTP error responses (400 vs 500) rather than silently succeeding
  with garbage data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Union

import pymupdf as fitz  # PyMuPDF's package is now named "pymupdf"; "fitz" is its legacy alias


class UnsupportedFileTypeError(Exception):
    """Raised when a file extension has no registered loader."""


class EmptyDocumentError(Exception):
    """Raised when a file contains no extractable text at all."""


class CorruptedDocumentError(Exception):
    """Raised when a file exists but cannot be parsed (e.g. broken PDF)."""


@dataclass
class PageContent:
    """Text extracted from a single logical page.

    For PDFs, page_number is the real 1-indexed PDF page number (used
    later for source citations like "Page 12"). For TXT/MD, the whole
    file is page_number=1 since those formats have no page concept.
    """

    page_number: int
    text: str


@dataclass
class LoadedDocument:
    """Result of loading one document: text + metadata, nothing else."""

    filename: str
    file_type: str  # "pdf" | "txt" | "md" (extension without the dot)
    pages: List[PageContent] = field(default_factory=list)
    full_text: str = ""
    char_count: int = 0
    page_count: int = 0


class DocumentLoader:
    """Loads text from supported file types.

    To support a new file type, add one entry to `self._loaders`
    mapping the extension to a function with signature
    `(Path) -> List[PageContent]`. Nothing else in the pipeline needs
    to change.
    """

    def __init__(self) -> None:
        self._loaders: Dict[str, Callable[[Path], List[PageContent]]] = {
            ".pdf": self._load_pdf,
            ".txt": self._load_plain_text,
            ".md": self._load_plain_text,
        }

    def supported_extensions(self) -> List[str]:
        """Extensions this loader currently knows how to handle."""
        return list(self._loaders.keys())

    def load(self, file_path: Union[str, Path]) -> LoadedDocument:
        """Load a document from disk and return its text + metadata.

        Raises:
            FileNotFoundError: path does not exist.
            UnsupportedFileTypeError: extension has no registered loader.
            CorruptedDocumentError: file exists but could not be parsed.
            EmptyDocumentError: file parsed fine but has no usable text
                (e.g. a scanned/image-only PDF with no text layer).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        loader_fn = self._loaders.get(ext)
        if loader_fn is None:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. "
                f"Supported types: {self.supported_extensions()}"
            )

        pages = loader_fn(path)
        # Drop pages that are blank after normalization (common in PDFs
        # with cover pages, section dividers, or scanned blank pages).
        non_empty_pages = [p for p in pages if p.text.strip()]

        if not non_empty_pages:
            raise EmptyDocumentError(
                f"No extractable text found in '{path.name}'. "
                "If this is a scanned PDF, OCR would be required "
                "(see README 'Limitations')."
            )

        full_text = "\n\n".join(p.text for p in non_empty_pages)
        return LoadedDocument(
            filename=path.name,
            file_type=ext.lstrip("."),
            pages=non_empty_pages,
            full_text=full_text,
            char_count=len(full_text),
            page_count=len(non_empty_pages),
        )

    # -- individual format loaders --------------------------------------

    @staticmethod
    def _load_pdf(path: Path) -> List[PageContent]:
        """Extract text page-by-page from a PDF using PyMuPDF.

        Page numbers are preserved exactly as they appear in the PDF
        (1-indexed) so later source citations ("Page 12") are accurate
        and never fabricated.
        """
        try:
            doc = fitz.open(path)
        except Exception as exc:  # PyMuPDF raises its own exception types
            raise CorruptedDocumentError(
                f"Failed to open PDF '{path.name}': {exc}"
            ) from exc

        pages: List[PageContent] = []
        try:
            for index, page in enumerate(doc):
                raw_text = page.get_text("text")
                pages.append(
                    PageContent(
                        page_number=index + 1,
                        text=DocumentLoader._normalize_text(raw_text),
                    )
                )
        finally:
            doc.close()

        return pages

    @staticmethod
    def _load_plain_text(path: Path) -> List[PageContent]:
        """Read a .txt or .md file as a single logical page.

        Tries UTF-8 first (the overwhelming common case) and falls
        back to Latin-1 so we don't crash on files with unusual
        encodings -- Latin-1 can decode any byte sequence, which is
        exactly what we want as a last-resort fallback here.
        """
        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = path.read_text(encoding="latin-1")

        text = DocumentLoader._normalize_text(raw_text)
        return [PageContent(page_number=1, text=text)]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Light, format-preserving cleanup applied to all extracted text.

        - Normalizes Windows/old-Mac line endings to \\n.
        - Collapses runs of spaces/tabs (common PDF extraction artifact
          where justified text produces irregular spacing).
        - Collapses 3+ consecutive newlines down to exactly 2, which
          keeps paragraph boundaries (used later by the chunker) intact
          without letting huge blank gaps inflate character counts.
        This intentionally does NOT strip punctuation, lowercase text,
        or remove stopwords -- that kind of aggressive normalization
        would hurt both citation readability and embedding quality.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
