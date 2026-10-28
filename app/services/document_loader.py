"""
app/services/document_loader.py
─────────────────────────────────────────────────────────────────────────────
Loads raw text from supported document formats:

    • .txt  — plain text (UTF-8 with BOM tolerance)
    • .md   — Markdown (treated as plain text; headings preserved)
    • .pdf  — PDF (via pypdf; page numbers recorded)
    • .docx — Word (via python-docx; headings detected)

Returns a list of PageBlock objects that carry the raw text of each logical
"page" or section together with metadata that the chunker will propagate into
every chunk it produces.

Design rules
────────────
1. Never silently discard text.  Log extraction errors and include a
   placeholder block so downstream code knows a page failed.
2. Keep newlines inside blocks; the chunker handles whitespace normalisation.
3. Detect scanned PDFs (pages with <50 chars of extracted text) and warn —
   the loader does NOT do OCR itself but marks blocks with needs_ocr=True so
   an optional OCR adapter can post-process them.
4. Repeated boilerplate (running headers/footers) is NOT stripped at load
   time; the chunker handles deduplication so the loader stays simple.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Optional heavy imports ────────────────────────────────────────────────────

try:
    from pypdf import PdfReader
    _PDF_OK = True
except ImportError:
    _PDF_OK = False
    logger.warning("pypdf not installed — PDF loading disabled. Run: pip install pypdf")

try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False
    logger.warning("python-docx not installed — DOCX loading disabled. Run: pip install python-docx")


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class PageBlock:
    """
    One logical "page" or section extracted from a source document.
    Multiple PageBlocks are produced per document (one per PDF page,
    one per DOCX heading section, or one for the whole .txt/.md file).
    """
    text: str                          # Raw extracted text, newlines preserved
    page_number: Optional[int] = None  # 1-based page index (PDF/DOCX)
    section_title: str = ""            # Nearest heading above this block
    source_path: str = ""              # Absolute or repo-relative path
    document_title: str = ""           # Derived from filename or first heading
    needs_ocr: bool = False            # True for scanned PDF pages
    extra: dict = field(default_factory=dict)   # Any loader-specific metadata


@dataclass
class LoadedDocument:
    """Aggregates all PageBlocks for one source file."""
    source_path: str
    document_title: str
    format: str                        # "txt" | "md" | "pdf" | "docx"
    blocks: list[PageBlock]
    total_pages: int = 0
    error: Optional[str] = None        # Non-None if the file failed to load


# ── Public entry point ────────────────────────────────────────────────────────

def load_document(path: str | Path) -> LoadedDocument:
    """
    Dispatch to the correct format loader and return a LoadedDocument.
    Raises ValueError for unsupported extensions.
    """
    p = Path(path).resolve()
    ext = p.suffix.lower()

    if not p.exists():
        return LoadedDocument(
            source_path=str(p),
            document_title=p.stem,
            format=ext.lstrip("."),
            blocks=[],
            error=f"File not found: {p}",
        )

    loaders = {
        ".txt": _load_txt,
        ".md":  _load_md,
        ".pdf": _load_pdf,
        ".docx": _load_docx,
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported document format: '{ext}'. "
            f"Supported formats: {list(loaders.keys())}"
        )

    try:
        doc = loader(p)
        logger.info(
            "Loaded %s (%d blocks, %d bytes)",
            p.name, len(doc.blocks), p.stat().st_size
        )
        return doc
    except Exception as exc:
        logger.exception("Failed to load %s: %s", p.name, exc)
        return LoadedDocument(
            source_path=str(p),
            document_title=p.stem,
            format=ext.lstrip("."),
            blocks=[],
            error=str(exc),
        )


# ── Format-specific loaders ───────────────────────────────────────────────────

def _derive_title(path: Path, text: str) -> str:
    """
    Try to extract a meaningful title:
    1. First # heading in the text.
    2. Fallback: filename stem formatted as title-case words.
    """
    heading_match = re.search(r"^#+\s+(.+)$", text, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def _load_txt(path: Path) -> LoadedDocument:
    """Load plain text as a single block."""
    text = path.read_text(encoding="utf-8-sig")   # utf-8-sig handles BOM
    title = _derive_title(path, text)
    block = PageBlock(
        text=text,
        page_number=1,
        section_title="",
        source_path=str(path),
        document_title=title,
    )
    return LoadedDocument(
        source_path=str(path),
        document_title=title,
        format="txt",
        blocks=[block],
        total_pages=1,
    )


def _load_md(path: Path) -> LoadedDocument:
    """
    Load Markdown, splitting at level-1 and level-2 headings.
    Each heading section becomes its own PageBlock so downstream
    chunking keeps heading context inside every chunk.
    """
    raw = path.read_text(encoding="utf-8-sig")
    title = _derive_title(path, raw)

    # Split on H1/H2 headings
    heading_pattern = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []   # (heading_text, body)

    last_end = 0
    preamble = ""
    current_heading = ""

    for match in heading_pattern.finditer(raw):
        body = raw[last_end:match.start()].strip()
        if last_end == 0:
            preamble = body
        else:
            sections.append((current_heading, body))
        current_heading = match.group(2).strip()
        last_end = match.end()

    # Final section
    body = raw[last_end:].strip()
    if current_heading:
        sections.append((current_heading, body))
    elif preamble:
        sections.append(("", preamble))

    if not sections:
        sections = [("", raw)]

    blocks: list[PageBlock] = []
    for idx, (heading, body) in enumerate(sections, start=1):
        if not body.strip():
            continue
        full_text = f"{heading}\n\n{body}" if heading else body
        blocks.append(PageBlock(
            text=full_text,
            page_number=idx,
            section_title=heading,
            source_path=str(path),
            document_title=title,
        ))

    return LoadedDocument(
        source_path=str(path),
        document_title=title,
        format="md",
        blocks=blocks,
        total_pages=len(blocks),
    )


_SCANNED_THRESHOLD = 50   # characters per page to flag as potentially scanned


def _load_pdf(path: Path) -> LoadedDocument:
    """
    Load each PDF page as a separate PageBlock.
    Pages with very little text are flagged as needs_ocr=True.
    """
    if not _PDF_OK:
        return LoadedDocument(
            source_path=str(path),
            document_title=path.stem,
            format="pdf",
            blocks=[],
            error="pypdf not installed — cannot load PDF",
        )

    reader = PdfReader(str(path))
    title = path.stem.replace("_", " ").replace("-", " ").title()
    blocks: list[PageBlock] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        needs_ocr = len(text.strip()) < _SCANNED_THRESHOLD

        if needs_ocr:
            logger.warning(
                "PDF page %d of '%s' has very little text (%d chars) — "
                "may be a scanned page requiring OCR.",
                page_idx, path.name, len(text.strip())
            )

        blocks.append(PageBlock(
            text=text,
            page_number=page_idx,
            section_title="",
            source_path=str(path),
            document_title=title,
            needs_ocr=needs_ocr,
        ))

    return LoadedDocument(
        source_path=str(path),
        document_title=title,
        format="pdf",
        blocks=blocks,
        total_pages=len(blocks),
    )


def _load_docx(path: Path) -> LoadedDocument:
    """
    Load DOCX by iterating paragraphs.
    Heading styles (Heading 1 / Heading 2) split the document into sections.
    Each section becomes its own PageBlock.
    """
    if not _DOCX_OK:
        return LoadedDocument(
            source_path=str(path),
            document_title=path.stem,
            format="docx",
            blocks=[],
            error="python-docx not installed — cannot load DOCX",
        )

    doc = DocxDocument(str(path))
    title = path.stem.replace("_", " ").replace("-", " ").title()

    # First pass: collect heading structure
    sections: list[dict] = []   # {heading, lines}
    current_heading = ""
    current_lines: list[str] = []

    heading_styles = {"heading 1", "heading 2", "heading 3", "title"}

    for para in doc.paragraphs:
        style_name = (para.style.name or "").lower()
        text = para.text.strip()
        if not text:
            continue

        if any(hs in style_name for hs in heading_styles):
            # Flush previous section
            if current_lines:
                sections.append({"heading": current_heading, "lines": current_lines})
            current_heading = text
            current_lines = []
            # Detect document title from first Title / Heading 1
            if not title and "title" in style_name:
                title = text
        else:
            current_lines.append(text)

    # Flush last section
    if current_lines or current_heading:
        sections.append({"heading": current_heading, "lines": current_lines})

    if not sections:
        # Fallback: whole document as one block
        full_text = "\n".join(
            p.text for p in doc.paragraphs if p.text.strip()
        )
        sections = [{"heading": "", "lines": [full_text]}]

    blocks: list[PageBlock] = []
    for idx, section in enumerate(sections, start=1):
        body = "\n".join(section["lines"])
        heading = section["heading"]
        full_text = f"{heading}\n\n{body}" if heading else body
        if not full_text.strip():
            continue
        blocks.append(PageBlock(
            text=full_text,
            page_number=idx,
            section_title=heading,
            source_path=str(path),
            document_title=title or path.stem,
        ))

    return LoadedDocument(
        source_path=str(path),
        document_title=title or path.stem,
        format="docx",
        blocks=blocks,
        total_pages=len(blocks),
    )


# ── Batch loader ──────────────────────────────────────────────────────────────

def load_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx"),
    recursive: bool = True,
) -> list[LoadedDocument]:
    """
    Load every document of the given types from a directory.
    Returns list of LoadedDocument (including failed ones — check .error).
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    pattern = "**/*" if recursive else "*"
    docs: list[LoadedDocument] = []

    for ext in extensions:
        for file_path in root.glob(f"{pattern}{ext}"):
            docs.append(load_document(file_path))

    logger.info(
        "Batch load: %d documents from '%s' (%d failed)",
        len(docs),
        root,
        sum(1 for d in docs if d.error),
    )
    return docs
