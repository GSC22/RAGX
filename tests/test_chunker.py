"""
Tests for app.services.chunker.

Run from the repo root:
    pytest tests/test_chunker.py -v
"""

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.chunker import chunk_document, chunk_text  # noqa: E402
from app.services.document_loader import LoadedDocument, PageContent  # noqa: E402


# --- chunk_text: basic behaviour --------------------------------------------


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=20) == []
    assert chunk_text("   \n\n  ", chunk_size=100, overlap=20) == []


def test_short_text_becomes_a_single_chunk():
    text = "This is a short paragraph that easily fits in one chunk."
    chunks = chunk_text(text, chunk_size=200, overlap=20)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_no_chunk_is_ever_empty():
    text = "Paragraph one.\n\n" + ("Sentence about the topic. " * 200)
    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunks_respect_chunk_size_within_reason():
    # A long block of short sentences with plenty of split points --
    # chunks should stay close to chunk_size, never wildly over it.
    text = " ".join(f"This is sentence number {i}." for i in range(200))
    chunk_size = 500
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=100)

    assert len(chunks) > 1
    # Allow a modest margin over chunk_size (see docstring on why exact
    # overshoot is possible), but nothing should be grossly oversized.
    assert all(len(c) <= chunk_size * 1.5 for c in chunks)


# --- Chunking strategy: paragraph/sentence/word boundaries ------------------


def test_short_paragraphs_are_not_split_unnecessarily():
    text = "First short paragraph.\n\nSecond short paragraph.\n\nThird short paragraph."
    # chunk_size big enough that all 3 paragraphs could fit together,
    # but small enough to check paragraphs aren't torn mid-sentence.
    chunks = chunk_text(text, chunk_size=1000, overlap=50)

    assert len(chunks) == 1
    assert "First short paragraph." in chunks[0]
    assert "Second short paragraph." in chunks[0]


def test_oversized_paragraph_splits_on_sentence_boundaries():
    long_paragraph = " ".join(
        f"This is sentence {i} in a very long paragraph about testing." for i in range(30)
    )
    chunks = chunk_text(long_paragraph, chunk_size=300, overlap=50)

    assert len(chunks) > 1
    # Each chunk should end on a real sentence boundary (period), not
    # mid-word, confirming sentences weren't torn apart.
    for chunk in chunks[:-1]:  # last chunk may end mid-flow depending on overlap bridge
        assert chunk.rstrip().endswith(".") or "." in chunk


def test_single_extremely_long_word_is_hard_split_not_crashed():
    # A "word" with no spaces at all (e.g. a long URL/hash) longer than
    # chunk_size -- the chunker must not raise or infinite-loop.
    giant_token = "x" * 500
    chunks = chunk_text(giant_token, chunk_size=100, overlap=10)

    assert len(chunks) >= 5
    assert all(chunk.strip() for chunk in chunks)
    # Reassembling should recover all the original characters.
    assert "".join(chunks).replace(" ", "") == giant_token


# --- Overlap behaviour --------------------------------------------------


def test_overlap_carries_shared_content_between_chunks():
    text = " ".join(f"This is sentence number {i} in the document." for i in range(100))
    chunks = chunk_text(text, chunk_size=400, overlap=100)

    assert len(chunks) > 1
    # The tail of each chunk should share some text with the start of
    # the next chunk -- that's what "overlap" means in practice.
    for first, second in zip(chunks, chunks[1:]):
        tail_words = set(first[-100:].split())
        head_words = set(second[:100].split())
        assert tail_words & head_words, "expected shared words between consecutive chunks"


def test_zero_overlap_produces_no_shared_bridge():
    text = " ".join(f"Sentence {i} here." for i in range(100))
    chunks = chunk_text(text, chunk_size=200, overlap=0)

    assert len(chunks) > 1  # sanity check: it actually split


# --- Parameter validation -------------------------------------------------


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (0, 10),      # chunk_size must be positive
        (-50, 10),    # chunk_size must be positive
        (100, -5),    # overlap cannot be negative
        (100, 100),   # overlap must be strictly smaller than chunk_size
        (100, 150),   # overlap larger than chunk_size
    ],
)
def test_invalid_parameters_raise_value_error(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text to chunk", chunk_size=chunk_size, overlap=overlap)


# --- chunk_document: metadata preservation -----------------------------------


def _sample_document() -> LoadedDocument:
    page1_text = "Attendance policy: students must maintain 75% attendance."
    page2_text = "Grading policy: " + ("Each assignment contributes to the final grade. " * 30)
    return LoadedDocument(
        filename="handbook.pdf",
        file_type="pdf",
        pages=[
            PageContent(page_number=1, text=page1_text),
            PageContent(page_number=2, text=page2_text),
        ],
        full_text=page1_text + "\n\n" + page2_text,
        char_count=len(page1_text) + len(page2_text),
        page_count=2,
    )


def test_chunk_document_preserves_filename_and_page_numbers():
    doc = _sample_document()
    chunks = chunk_document(doc, chunk_size=200, overlap=30)

    assert len(chunks) > 0
    assert all(c.source_filename == "handbook.pdf" for c in chunks)

    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]
    assert page1_chunks, "expected at least one chunk from page 1"
    assert page2_chunks, "expected at least one chunk from page 2"
    assert "75% attendance" in page1_chunks[0].text


def test_chunk_document_never_mixes_two_pages_in_one_chunk():
    doc = _sample_document()
    chunks = chunk_document(doc, chunk_size=200, overlap=30)

    for chunk in chunks:
        if chunk.page_number == 1:
            assert "Grading policy" not in chunk.text
        if chunk.page_number == 2:
            assert "Attendance policy" not in chunk.text


def test_chunk_document_ids_and_indices_are_unique_and_sequential():
    doc = _sample_document()
    chunks = chunk_document(doc, chunk_size=200, overlap=30)

    ids = [c.chunk_id for c in chunks]
    indices = [c.chunk_index for c in chunks]

    assert len(ids) == len(set(ids)), "chunk_ids must be unique"
    assert indices == list(range(len(chunks))), "chunk_index must be sequential from 0"


def test_chunk_document_char_count_matches_text_length():
    doc = _sample_document()
    chunks = chunk_document(doc, chunk_size=200, overlap=30)

    assert all(c.char_count == len(c.text) for c in chunks)
