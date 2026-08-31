"""
Chunking service.

Splits extracted document text into overlapping chunks suitable for
embedding and retrieval. Written entirely by hand (no LangChain /
LlamaIndex text splitters) so every decision here is explainable.

Two-phase strategy:

  Phase A (`_split_into_segments`): break the text into "segments" that
  are each already <= chunk_size, preferring to cut at the most natural
  boundary available -- paragraph, then sentence, then word, and only
  as an absolute last resort (a single "word" longer than chunk_size,
  e.g. a long URL) a hard character cut. This phase never merges
  anything; it only breaks things down.

  Phase B (`_merge_segments_with_overlap`): greedily pack those segments
  back together into chunks up to chunk_size, and carry the trailing
  `overlap` characters of one chunk into the start of the next so a
  fact sitting right on a chunk boundary still appears whole in at
  least one chunk.

Chunking is done per-page (see `chunk_document`), not on the whole
document as one string, so every chunk can be attributed to exactly
one page number for accurate citations later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.services.document_loader import LoadedDocument


@dataclass
class DocumentChunk:
    """A single chunk of text plus everything needed to cite it later."""

    chunk_id: str
    chunk_index: int  # position of this chunk within the whole document, 0-indexed
    source_filename: str
    page_number: int
    text: str
    char_count: int


# Sentence boundary: a '.', '!' or '?' followed by whitespace. Using a
# lookbehind keeps the punctuation attached to the sentence it ends,
# rather than stripping it into a separate token.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a single string of text into overlapping chunks.

    Args:
        text: the text to split (already extracted/normalized).
        chunk_size: maximum characters per chunk (soft limit -- see
            docstring for _merge_segments_with_overlap for the one
            case it can be exceeded).
        overlap: characters of trailing context carried from the end
            of one chunk into the start of the next.

    Returns:
        List of chunk strings, in order. Empty list for empty/whitespace
        -only input -- we never produce empty chunks.

    Raises:
        ValueError: for nonsensical configuration (chunk_size <= 0,
            overlap < 0, or overlap >= chunk_size -- the last one would
            mean each new chunk starts before the previous one ended
            by more than a full chunk, which can loop indefinitely).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    segments = _split_into_segments(text, chunk_size)
    return _merge_segments_with_overlap(segments, chunk_size, overlap)


def chunk_document(
    document: LoadedDocument, chunk_size: int, overlap: int
) -> List[DocumentChunk]:
    """Chunk every page of a loaded document, preserving page metadata.

    Each page is chunked independently (see module docstring for why),
    so a returned chunk's `page_number` is always exact -- never
    approximated or guessed.
    """
    chunks: List[DocumentChunk] = []
    global_index = 0

    for page in document.pages:
        for piece in chunk_text(page.text, chunk_size, overlap):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.filename}::p{page.page_number}::c{global_index}",
                    chunk_index=global_index,
                    source_filename=document.filename,
                    page_number=page.page_number,
                    text=piece,
                    char_count=len(piece),
                )
            )
            global_index += 1

    return chunks


# --- Phase A: break text down into segments each <= chunk_size -------------


def _split_into_segments(text: str, chunk_size: int) -> List[str]:
    """Split text into paragraphs, recursing into sentence/word splits
    only for paragraphs that individually exceed chunk_size.

    A paragraph that already fits is never touched further -- this is
    what keeps short-to-medium paragraphs intact in a single chunk
    instead of being needlessly fragmented.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    segments: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            segments.append(paragraph)
        else:
            segments.extend(_split_by_sentence(paragraph, chunk_size))
    return segments


def _split_by_sentence(text: str, chunk_size: int) -> List[str]:
    """Split an over-long paragraph into sentences, recursing into
    word-level splitting only for sentences that are themselves too long
    (e.g. a run-on sentence, or a paragraph with no real punctuation)."""
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]

    segments: List[str] = []
    for sentence in sentences:
        if len(sentence) <= chunk_size:
            segments.append(sentence)
        else:
            segments.extend(_split_by_words(sentence, chunk_size))
    return segments


def _split_by_words(text: str, chunk_size: int) -> List[str]:
    """Pack words greedily up to chunk_size. Last-resort fallback for
    text with no paragraph or sentence boundaries to split on.

    A single "word" longer than chunk_size (a long URL or hash, say)
    cannot be split on a natural boundary at all -- we hard-cut it by
    character count rather than raising, since refusing to chunk a
    document just because it contains one long token would be worse.
    """
    words = text.split(" ")
    segments: List[str] = []
    current_words: List[str] = []
    current_len = 0

    for word in words:
        if len(word) > chunk_size:
            # Flush whatever we've accumulated, then hard-cut this word.
            if current_words:
                segments.append(" ".join(current_words))
                current_words, current_len = [], 0
            for i in range(0, len(word), chunk_size):
                segments.append(word[i : i + chunk_size])
            continue

        added_len = len(word) + (1 if current_words else 0)  # +1 for the joining space
        if current_len + added_len > chunk_size and current_words:
            segments.append(" ".join(current_words))
            current_words, current_len = [word], len(word)
        else:
            current_words.append(word)
            current_len += added_len

    if current_words:
        segments.append(" ".join(current_words))
    return segments


# --- Phase B: merge segments back into overlapping chunks -----------------


def _merge_segments_with_overlap(
    segments: List[str], chunk_size: int, overlap: int
) -> List[str]:
    """Greedily pack segments into chunks up to chunk_size, carrying
    `overlap` characters of context from the end of one chunk into the
    start of the next.

    Note on the one case this can slightly exceed chunk_size: if the
    overlap text plus the next segment together are still bigger than
    chunk_size, we let that single chunk run a bit long rather than
    truncating mid-segment -- cutting a segment we just worked to keep
    intact would defeat the point of Phase A. In practice, since
    overlap is a fraction of chunk_size (150/800 by default) and
    segments are individually <= chunk_size, the overshoot is small.
    """
    if not segments:
        return []

    chunks: List[str] = []
    current = segments[0]

    for segment in segments[1:]:
        candidate = f"{current} {segment}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # Current chunk is full -- finalize it and start the next one,
        # carrying over the trailing `overlap` characters as a "bridge"
        # so content near the boundary isn't lost from every chunk.
        chunks.append(current)

        # Shrink the overlap if the next segment is large, so the two
        # combined don't balloon far past chunk_size for no reason.
        room_for_overlap = max(0, chunk_size - len(segment))
        effective_overlap = min(overlap, room_for_overlap, len(current))

        bridge = current[-effective_overlap:] if effective_overlap > 0 else ""
        current = f"{bridge} {segment}".strip() if bridge else segment

    chunks.append(current)
    return chunks
