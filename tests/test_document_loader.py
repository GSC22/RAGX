"""
Tests for app.services.document_loader.

Run from the `backend/` directory (or with backend on PYTHONPATH):
    pytest ../tests/test_document_loader.py -v
"""

import sys
from pathlib import Path

import fitz  # PyMuPDF, also used here to *generate* a test PDF
import pytest

# Make `backend/app` importable when running pytest from the repo root.
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.document_loader import (  # noqa: E402
    DocumentLoader,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


def _make_pdf(path: Path, pages_text: list[str]) -> None:
    """Helper: build a small real PDF with one page per string, so tests
    don't depend on a fixture file living in the repo."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


# --- TXT / MD -------------------------------------------------------------


def test_load_txt_file(tmp_path: Path, loader: DocumentLoader):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("This is a simple text document.\n\nIt has two paragraphs.")

    result = loader.load(file_path)

    assert result.filename == "notes.txt"
    assert result.file_type == "txt"
    assert result.page_count == 1
    assert result.pages[0].page_number == 1
    assert "simple text document" in result.full_text
    assert result.char_count == len(result.full_text)


def test_load_md_file(tmp_path: Path, loader: DocumentLoader):
    file_path = tmp_path / "readme.md"
    file_path.write_text("# Heading\n\nSome **markdown** content.")

    result = loader.load(file_path)

    assert result.file_type == "md"
    assert "Heading" in result.full_text
    assert "markdown" in result.full_text


def test_txt_normalizes_excess_whitespace(tmp_path: Path, loader: DocumentLoader):
    file_path = tmp_path / "messy.txt"
    file_path.write_text("Paragraph one.\n\n\n\n\nParagraph   two   has  extra spaces.")

    result = loader.load(file_path)

    # 3+ newlines collapse to exactly 2 (a single paragraph break).
    assert "\n\n\n" not in result.full_text
    # Runs of spaces/tabs collapse to a single space.
    assert "   " not in result.full_text


# --- PDF --------------------------------------------------------------------


def test_load_pdf_extracts_text_per_page(tmp_path: Path, loader: DocumentLoader):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["Page one content about attendance policy.", "Page two content about grading."])

    result = loader.load(pdf_path)

    assert result.file_type == "pdf"
    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert "attendance policy" in result.pages[0].text
    assert "grading" in result.pages[1].text


def test_load_pdf_skips_blank_pages(tmp_path: Path, loader: DocumentLoader):
    pdf_path = tmp_path / "with_blank.pdf"
    # Middle page intentionally left blank (e.g. a section divider).
    _make_pdf(pdf_path, ["First real page.", "", "Third real page."])

    result = loader.load(pdf_path)

    # Only the two non-blank pages should survive, and original page
    # numbers must be preserved (1 and 3), not renumbered to (1 and 2) --
    # renumbering would produce incorrect citations later.
    assert result.page_count == 2
    assert [p.page_number for p in result.pages] == [1, 3]


# --- Error handling -----------------------------------------------------


def test_unsupported_extension_raises(tmp_path: Path, loader: DocumentLoader):
    file_path = tmp_path / "data.csv"
    file_path.write_text("a,b,c\n1,2,3")

    with pytest.raises(UnsupportedFileTypeError):
        loader.load(file_path)


def test_missing_file_raises(loader: DocumentLoader):
    with pytest.raises(FileNotFoundError):
        loader.load("/tmp/this_file_does_not_exist_at_all.txt")


def test_empty_text_file_raises(tmp_path: Path, loader: DocumentLoader):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   \n\n   ")  # whitespace only

    with pytest.raises(EmptyDocumentError):
        loader.load(file_path)


def test_supported_extensions_lists_all_three(loader: DocumentLoader):
    exts = loader.supported_extensions()
    assert set(exts) == {".pdf", ".txt", ".md"}
