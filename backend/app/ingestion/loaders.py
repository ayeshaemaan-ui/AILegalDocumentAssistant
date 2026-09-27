"""Document processing (Tier 1 M1: Document Processing).

Turns a PDF / TXT / DOCX into a `LoadedDocument`: normalised text plus a page
map, so every answer can point back at a real location in a real file.

On page numbers, honestly: PDFs give exact ones (pdfplumber knows the page
boundaries). The CUAD .txt extracts do not - only about two thirds of them
carry a recoverable page sequence. Rather than invent numbers and present them
as fact, each page carries a `confidence` flag, and the UI labels an estimated
page as approximate. The primary locator is the section path ("Section 12.3"),
which is both reliable and what a lawyer would actually cite.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# A run of >=3 newlines is how the SEC text extracts separate printed pages.
PAGE_BREAK_RE = re.compile(r"\s*\n{3,}\s*")
TRAILING_PAGE_NUM_RE = re.compile(r"(\d{1,4})\s*$")
CHARS_PER_PAGE_ESTIMATE = 3200


@dataclass(slots=True)
class PageText:
    page_number: int
    text: str
    char_start: int
    char_end: int
    confidence: Literal["exact", "recovered", "estimated"] = "estimated"


@dataclass(slots=True)
class LoadedDocument:
    doc_id: str
    title: str
    source_path: str
    file_type: str
    full_text: str
    pages: list[PageText] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def page_for_offset(self, offset: int) -> PageText | None:
        for page in self.pages:
            if page.char_start <= offset < page.char_end:
                return page
        return self.pages[-1] if self.pages else None


def make_doc_id(path: Path | str, content: str) -> str:
    """Content-addressed ID: re-uploading the same file cannot duplicate it."""
    digest = hashlib.sha256(content.encode("utf-8", "ignore")).hexdigest()[:12]
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", Path(path).stem)[:60].strip("_").lower()
    return f"{stem}__{digest}"


def normalise_text(text: str) -> str:
    """Collapse the artefacts of PDF-to-text extraction.

    Kept deliberately conservative: we fix whitespace and ligatures but never
    drop characters, because a mangled clause is worse than an ugly one.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = text.replace(" ", " ").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # De-hyphenate words broken across a line break ("indemni-\nfication").
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


# --------------------------------------------------------------------- TXT
def _recover_txt_pages(text: str) -> list[PageText]:
    """Try to recover printed page numbers; fall back to a length estimate.

    Only accepted when the recovered numbers form a genuinely increasing run -
    many extracts repeat an exhibit number at every break, which would otherwise
    look like "page 18" on all 60 chunks.
    """
    segments: list[tuple[str, int, int]] = []
    cursor = 0
    for match in PAGE_BREAK_RE.finditer(text):
        segments.append((text[cursor : match.start()], cursor, match.start()))
        cursor = match.end()
    segments.append((text[cursor:], cursor, len(text)))

    recovered: list[int | None] = []
    for body, _, _ in segments[:-1]:
        m = TRAILING_PAGE_NUM_RE.search(body.rstrip())
        recovered.append(int(m.group(1)) if m else None)

    numbers = [n for n in recovered if n is not None]
    increasing = sum(1 for a, b in zip(numbers, numbers[1:]) if b == a + 1)
    trustworthy = len(numbers) >= 3 and increasing >= max(2, len(numbers) // 3)

    if trustworthy:
        pages: list[PageText] = []
        current = numbers[0] if numbers else 1
        for idx, (body, start, end) in enumerate(segments):
            num = recovered[idx] if idx < len(recovered) and recovered[idx] else current
            pages.append(PageText(num, body, start, end, confidence="recovered"))
            current = num + 1
        return pages

    return _estimate_pages(text)


def _estimate_pages(text: str) -> list[PageText]:
    pages: list[PageText] = []
    for idx, start in enumerate(range(0, len(text), CHARS_PER_PAGE_ESTIMATE)):
        end = min(start + CHARS_PER_PAGE_ESTIMATE, len(text))
        pages.append(PageText(idx + 1, text[start:end], start, end, confidence="estimated"))
    return pages or [PageText(1, text, 0, len(text), confidence="estimated")]


def load_txt(path: Path) -> LoadedDocument:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    text = normalise_text(raw)
    return LoadedDocument(
        doc_id=make_doc_id(path, text),
        title=Path(path).stem,
        source_path=str(path),
        file_type="txt",
        full_text=text,
        pages=_recover_txt_pages(text),
    )


# --------------------------------------------------------------------- PDF
def load_pdf(path: Path) -> LoadedDocument:
    """Exact page numbers - pdfplumber knows where each page starts and ends."""
    import pdfplumber

    pages: list[PageText] = []
    chunks: list[str] = []
    offset = 0
    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = normalise_text(page.extract_text() or "")
            if not page_text:
                continue
            block = page_text + "\n\n"
            chunks.append(block)
            pages.append(PageText(page_no, page_text, offset, offset + len(block), "exact"))
            offset += len(block)

    full_text = "".join(chunks).strip()
    if not full_text:  # scanned/image PDF with no text layer
        raise ValueError(
            f"No extractable text in {Path(path).name}. "
            "It is probably a scanned PDF and would need OCR."
        )
    return LoadedDocument(
        doc_id=make_doc_id(path, full_text),
        title=Path(path).stem,
        source_path=str(path),
        file_type="pdf",
        full_text=full_text,
        pages=pages,
    )


# -------------------------------------------------------------------- DOCX
def load_docx(path: Path) -> LoadedDocument:
    import docx

    document = docx.Document(str(path))
    text = normalise_text("\n".join(p.text for p in document.paragraphs))
    return LoadedDocument(
        doc_id=make_doc_id(path, text),
        title=Path(path).stem,
        source_path=str(path),
        file_type="docx",
        full_text=text,
        pages=_estimate_pages(text),
    )


LOADERS = {".txt": load_txt, ".pdf": load_pdf, ".docx": load_docx, ".md": load_txt}


def load_document(path: Path | str) -> LoadedDocument:
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported: {', '.join(sorted(LOADERS))}"
        )
    return loader(path)
