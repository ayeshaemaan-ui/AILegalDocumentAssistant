"""Chunking strategies (Module 5: Chunking Strategies + Tier 1 M1: Advanced Chunking).

Chunking is the highest-leverage decision in a RAG system. Retrieval can only
ever return a chunk, so if a chunk splits an obligation from its exception, no
amount of re-ranking or prompt engineering recovers the meaning.

Three strategies ship here so the evaluation harness can *prove* which is best
on CUAD ground truth rather than us asserting it:

  recursive         Baseline. Fixed-size windows on paragraph/sentence
                    boundaries. Fast, structure-blind.
  legal_structural  Default. Splits on the contract's own hierarchy
                    (ARTICLE > Section 5.2 > 5.2.1) so a chunk is a clause, and
                    carries the section path as metadata for citation and
                    filtering. Oversized sections are window-split with overlap;
                    tiny ones are merged into their neighbour.
  semantic          Splits where meaning shifts: embeds each sentence, and cuts
                    where the cosine distance between neighbours spikes past a
                    percentile threshold. Slowest (it embeds twice) but handles
                    unstructured documents with no numbering.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

from app.config import Settings, get_settings
from app.ingestion.loaders import LoadedDocument

logger = logging.getLogger(__name__)

# Heading forms that actually appear in commercial contracts, most specific
# first. Ordering matters: "5.2.1 Scope" must match before "5. DEFINITIONS".
HEADING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("article", re.compile(r"^\s*(ARTICLE|Article)\s+([IVXLC]+|\d+)\s*[.\-:]?\s*(.{0,90})$")),
    ("section", re.compile(r"^\s*(SECTION|Section)\s+(\d+(?:\.\d+)*)\s*[.\-:]?\s*(.{0,90})$")),
    ("numbered", re.compile(r"^\s*(\d+(?:\.\d+){0,4})\.?\s+([A-Z][^\n]{2,90})$")),
    ("lettered", re.compile(r"^\s*\(([a-z])\)\s+([A-Z][^\n]{2,90})$")),
    ("caps", re.compile(r"^\s*([A-Z][A-Z \-&',/]{5,70})\s*$")),
]

# The text extracts are mostly single-line, so headings also appear mid-line.
# This finds "  2.3.1. Opportunity. " style markers inside a flowing paragraph.
INLINE_SECTION_RE = re.compile(
    r"(?:(?<=\s)|^)((?:ARTICLE|Article|SECTION|Section)\s+(?:[IVXLC]+|\d+(?:\.\d+)*)|\d+(?:\.\d+){0,3})\.?\s+"
    r"(?=[A-Z][A-Za-z][^.]{2,80}[.:])"
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\"'])")


@dataclass(slots=True)
class Chunk:
    """One retrievable unit, plus everything needed to cite it."""

    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    strategy: str
    section_number: str | None = None
    section_path: str | None = None
    heading: str | None = None
    page_number: int | None = None
    page_confidence: str = "estimated"
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        """Flat dict for the vector store - Chroma rejects nested values."""
        base = {k: v for k, v in asdict(self).items() if k not in {"text", "metadata"}}
        base.update(self.metadata)
        return {k: v for k, v in base.items() if v is not None and not isinstance(v, (dict, list))}


# ---------------------------------------------------------------- helpers
def estimate_tokens(text: str) -> int:
    """Cheap heuristic (~4 chars/token) used only for budgeting and logging.

    Real counts come from Claude's `count_tokens` endpoint - never tiktoken,
    which is OpenAI's tokenizer and undercounts Claude by 15-20%.
    """
    return max(1, len(text) // 4)


def _window_split(text: str, start_offset: int, size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split oversized text on sentence boundaries with a sliding overlap.

    The overlap is what stops a cross-boundary sentence from becoming
    unretrievable: the clause's opening line appears in both neighbours.
    """
    sentences = SENTENCE_SPLIT_RE.split(text)
    out: list[tuple[str, int, int]] = []
    buf: list[str] = []
    buf_len = 0
    cursor = 0
    buf_start = 0

    for sentence in sentences:
        s_len = len(sentence) + 1
        if buf_len + s_len > size and buf:
            body = " ".join(buf).strip()
            out.append((body, start_offset + buf_start, start_offset + buf_start + len(body)))
            # Re-seed the buffer with the tail of the previous chunk.
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(buf):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            buf = tail
            buf_len = tail_len
            buf_start = max(0, cursor - tail_len)
        buf.append(sentence)
        buf_len += s_len
        cursor += s_len

    if buf:
        body = " ".join(buf).strip()
        if body:
            out.append((body, start_offset + buf_start, start_offset + buf_start + len(body)))
    return out


def _attach_page(chunk: Chunk, doc: LoadedDocument) -> Chunk:
    page = doc.page_for_offset(chunk.char_start)
    if page is not None:
        chunk.page_number = page.page_number
        chunk.page_confidence = page.confidence
    return chunk


# ------------------------------------------------------ strategy: recursive
def chunk_recursive(doc: LoadedDocument, settings: Settings) -> list[Chunk]:
    """Baseline: LangChain's RecursiveCharacterTextSplitter.

    Present so the evaluation has a fair control to beat, not because it is
    what we ship.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", ". ", "; ", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks: list[Chunk] = []
    for idx, piece in enumerate(splitter.create_documents([doc.full_text])):
        start = piece.metadata.get("start_index", 0)
        body = piece.page_content.strip()
        if len(body) < settings.min_chunk_chars:
            continue
        chunks.append(
            _attach_page(
                Chunk(
                    chunk_id=f"{doc.doc_id}::rec::{idx:05d}",
                    doc_id=doc.doc_id,
                    text=body,
                    char_start=start,
                    char_end=start + len(body),
                    strategy="recursive",
                    chunk_index=idx,
                ),
                doc,
            )
        )
    return chunks


# ----------------------------------------------- strategy: legal_structural
def _find_sections(text: str) -> list[tuple[int, str, str]]:
    """Return (offset, section_number, heading) for every detected heading."""
    found: list[tuple[int, str, str]] = []

    for line_match in re.finditer(r"^.*$", text, re.MULTILINE):
        line = line_match.group(0).strip()
        if not line or len(line) > 120:
            continue
        for kind, pattern in HEADING_PATTERNS:
            m = pattern.match(line)
            if not m:
                continue
            if kind in {"article", "section"}:
                number, heading = m.group(2), (m.group(3) or "").strip()
            elif kind == "caps":
                number, heading = "", m.group(1).strip()
            else:
                number, heading = m.group(1), m.group(2).strip()
            found.append((line_match.start(), number, heading))
            break

    for m in INLINE_SECTION_RE.finditer(text):
        number = m.group(1).strip()
        trailer = text[m.end() : m.end() + 90]
        heading = re.split(r"[.:]", trailer, maxsplit=1)[0].strip()
        found.append((m.start(), number, heading))

    # Dedupe near-identical hits from the two passes, keep document order.
    found.sort(key=lambda t: t[0])
    deduped: list[tuple[int, str, str]] = []
    for offset, number, heading in found:
        if deduped and offset - deduped[-1][0] < 5:
            continue
        deduped.append((offset, number, heading))
    return deduped


def _section_path(number: str, heading: str, stack: dict[int, str]) -> str:
    """Build "3 > 3.2 > 3.2.1" so a chunk knows its place in the hierarchy."""
    if not number:
        return heading or ""
    parts = number.split(".")
    depth = len(parts)
    stack[depth] = number
    for deeper in [d for d in stack if d > depth]:
        stack.pop(deeper, None)
    trail = [stack[d] for d in sorted(stack)]
    label = " > ".join(trail)
    return f"{label} {heading}".strip() if heading else label


def chunk_legal_structural(doc: LoadedDocument, settings: Settings) -> list[Chunk]:
    """Default strategy: one chunk per clause, with its section path attached."""
    text = doc.full_text
    sections = _find_sections(text)

    if len(sections) < 3:
        logger.debug("No usable structure in %s - falling back to recursive", doc.doc_id)
        fallback = chunk_recursive(doc, settings)
        for c in fallback:
            c.strategy = "legal_structural(fallback=recursive)"
        return fallback

    boundaries = [offset for offset, _, _ in sections]
    if boundaries[0] > 0:
        sections.insert(0, (0, "", "PREAMBLE"))
        boundaries.insert(0, 0)
    boundaries.append(len(text))

    stack: dict[int, str] = {}
    raw_units: list[Chunk] = []
    for idx, (offset, number, heading) in enumerate(sections):
        body = text[boundaries[idx] : boundaries[idx + 1]].strip()
        if not body:
            continue
        path = _section_path(number, heading, stack)
        # A clause bigger than the budget is window-split, but each piece keeps
        # the parent section path so citation survives the split.
        pieces = (
            _window_split(body, boundaries[idx], settings.chunk_size, settings.chunk_overlap)
            if len(body) > settings.chunk_size
            else [(body, boundaries[idx], boundaries[idx] + len(body))]
        )
        for piece_text, start, end in pieces:
            raw_units.append(
                Chunk(
                    chunk_id="",
                    doc_id=doc.doc_id,
                    text=piece_text,
                    char_start=start,
                    char_end=end,
                    strategy="legal_structural",
                    section_number=number or None,
                    section_path=path or None,
                    heading=heading or None,
                )
            )

    merged = _merge_small(raw_units, settings.min_chunk_chars, settings.chunk_size)
    for idx, chunk in enumerate(merged):
        chunk.chunk_id = f"{doc.doc_id}::sec::{idx:05d}"
        chunk.chunk_index = idx
        _attach_page(chunk, doc)
    return merged


def _merge_small(chunks: list[Chunk], min_chars: int, max_chars: int) -> list[Chunk]:
    """Fold sub-minimum fragments forward.

    Contracts are full of one-line headings ("12. MISCELLANEOUS"). Indexed
    alone they are noise that outranks real clauses on short queries.
    """
    out: list[Chunk] = []
    for chunk in chunks:
        if (
            out
            and len(chunk.text) < min_chars
            and len(out[-1].text) + len(chunk.text) < max_chars
            and out[-1].doc_id == chunk.doc_id
        ):
            prev = out[-1]
            prev.text = f"{prev.text}\n{chunk.text}".strip()
            prev.char_end = chunk.char_end
            continue
        out.append(chunk)
    return [c for c in out if len(c.text) >= min_chars // 2]


# ------------------------------------------------------- strategy: semantic
def chunk_semantic(doc: LoadedDocument, settings: Settings) -> list[Chunk]:
    """Cut where the topic actually changes, measured in embedding space."""
    import numpy as np

    from app.core.embeddings import get_embeddings

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(doc.full_text) if s.strip()]
    if len(sentences) < 4:
        return chunk_recursive(doc, settings)

    embedder = get_embeddings(settings)
    vectors = np.array(embedder.embed_documents(sentences))
    # Vectors are L2-normalised, so the dot product is the cosine similarity.
    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
    distances = 1.0 - similarities
    threshold = float(np.percentile(distances, settings.semantic_breakpoint_percentile))

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = 0
    cursor = 0
    idx = 0
    for i, sentence in enumerate(sentences):
        if not buf:
            buf_start = cursor
        buf.append(sentence)
        cursor += len(sentence) + 1
        at_breakpoint = i < len(distances) and distances[i] > threshold
        too_big = sum(len(s) for s in buf) > settings.chunk_size
        if (at_breakpoint and sum(len(s) for s in buf) >= settings.min_chunk_chars) or too_big:
            body = " ".join(buf)
            chunks.append(
                _attach_page(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::sem::{idx:05d}",
                        doc_id=doc.doc_id,
                        text=body,
                        char_start=buf_start,
                        char_end=buf_start + len(body),
                        strategy="semantic",
                        chunk_index=idx,
                    ),
                    doc,
                )
            )
            idx += 1
            buf = []

    if buf:
        body = " ".join(buf)
        chunks.append(
            _attach_page(
                Chunk(
                    chunk_id=f"{doc.doc_id}::sem::{idx:05d}",
                    doc_id=doc.doc_id,
                    text=body,
                    char_start=buf_start,
                    char_end=buf_start + len(body),
                    strategy="semantic",
                    chunk_index=idx,
                ),
                doc,
            )
        )
    return chunks


STRATEGIES: dict[str, Callable[[LoadedDocument, Settings], list[Chunk]]] = {
    "recursive": chunk_recursive,
    "legal_structural": chunk_legal_structural,
    "semantic": chunk_semantic,
}


def chunk_document(
    doc: LoadedDocument, settings: Settings | None = None, strategy: str | None = None
) -> list[Chunk]:
    settings = settings or get_settings()
    name = strategy or settings.chunk_strategy
    fn = STRATEGIES.get(name)
    if fn is None:
        raise ValueError(f"Unknown chunk strategy '{name}'. Options: {', '.join(STRATEGIES)}")
    return fn(doc, settings)
