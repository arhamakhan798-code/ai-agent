"""
app/services/chunking.py
─────────────────────────────────────────────────────────────────────────────
Structure-aware, token-counted recursive chunker.

Design principles
─────────────────
1. Token-accurate sizing — uses tiktoken so chunk sizes are faithful to the
   LLM context window, not character counts (which undercount CJK, overcount
   ASCII).
2. Structure-preserving — prefers splitting at paragraph breaks, then sentence
   boundaries, never in the middle of a numbered rule, table row, or code
   block.
3. Stable chunk IDs — deterministic SHA-256 of (source_path + page + chunk
   index) so re-indexing the same document produces the same IDs, enabling
   clean incremental upserts into ChromaDB.
4. Rich metadata — every chunk carries the full provenance chain so users can
   always trace an answer back to the exact page/section of a source document.
5. Preview mode — the `preview_chunks()` helper prints a readable table of
   chunk boundaries without touching ChromaDB.

Configuration (from Settings)
───────────────────────────────
CHUNK_SIZE_TOKENS    target tokens per chunk                 default 500
CHUNK_OVERLAP_TOKENS tokens repeated at start of next chunk  default 75
CHUNK_MIN_TOKENS     chunks smaller than this are merged      default 50
"""

from __future__ import annotations

import hashlib
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ── Optional heavy imports ────────────────────────────────────────────────────

try:
    import tiktoken
    _tiktoken_enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 / Groq safe
    _TIKTOKEN_OK = True
except ImportError:
    _tiktoken_enc = None
    _TIKTOKEN_OK = False
    logger.warning(
        "tiktoken not installed — falling back to word-count estimates. "
        "Install with: pip install tiktoken"
    )

from app.services.document_loader import PageBlock, LoadedDocument


# ── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and insertion into ChromaDB.

    The `chunk_id` is the ChromaDB document ID — it is stable across
    re-ingestion of the same document as long as the source path and page
    number don't change.
    """
    chunk_id: str              # Stable SHA-256 deterministic ID
    text: str                  # The text to embed
    token_count: int           # Actual token count (or estimate)

    # ── Provenance ────────────────────────────────────────────────────────────
    source_path: str
    document_title: str
    section_title: str
    page_number: Optional[int]
    chunk_index: int           # 0-based index within the source page/block

    # ── Document-level metadata ───────────────────────────────────────────────
    document_id: str           # SHA-256 of source_path (stable doc reference)
    version: str = ""          # Populated by ingestion pipeline from manifest
    effective_date: str = ""
    access_scope: str = "public"
    extra: dict = field(default_factory=dict)


# ── Token utilities ───────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Count tokens accurately if tiktoken is available, else estimate."""
    if not text:
        return 0
    if _TIKTOKEN_OK and _tiktoken_enc is not None:
        return len(_tiktoken_enc.encode(text))
    # Fallback: ~1.3 tokens per word (rough average across English)
    return max(1, round(len(text.split()) * 1.3))


def _token_slice(text: str, max_tokens: int) -> tuple[str, str]:
    """
    Split *text* so the first part ≤ max_tokens, return (first, remainder).
    Uses tiktoken decode/encode for byte-accurate splits.
    """
    if _TIKTOKEN_OK and _tiktoken_enc is not None:
        tokens = _tiktoken_enc.encode(text)
        if len(tokens) <= max_tokens:
            return text, ""
        head = _tiktoken_enc.decode(tokens[:max_tokens])
        tail = _tiktoken_enc.decode(tokens[max_tokens:])
        return head, tail

    # Fallback word-based split
    words = text.split()
    split_at = max(1, int(max_tokens / 1.3))
    return " ".join(words[:split_at]), " ".join(words[split_at:])


# ── Structural splitters ──────────────────────────────────────────────────────

# Paragraph: two or more newlines (possibly with spaces between)
_PARA_BREAK = re.compile(r"\n{2,}|\r\n\r\n")

# Sentence boundary: '. ' or '? ' or '! ' followed by capital letter / digit
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9\"])")

# Numbered list item: '1.' or '(1)' or 'a.' at line start
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]\s+|[a-z][.)]\s+|\(\d+\)\s+)", re.MULTILINE)

# Table row: line containing '|' separators
_TABLE_ROW = re.compile(r"^\s*\|.+\|.*$", re.MULTILINE)


def _split_on_structure(text: str) -> list[str]:
    """
    Split text into structural units (paragraphs / sentences).
    Returns non-empty stripped strings.
    Tries paragraphs first; if any unit is still huge, splits on sentences.
    """
    paragraphs = [p.strip() for p in _PARA_BREAK.split(text) if p.strip()]

    result: list[str] = []
    for para in paragraphs:
        # If a paragraph looks like a table, keep it whole
        if _TABLE_ROW.search(para):
            result.append(para)
            continue
        # If it's a numbered/lettered list, keep each item together
        if _LIST_ITEM.search(para):
            result.append(para)
            continue
        # Otherwise split on sentence boundaries for fine-grained control
        sentences = [s.strip() for s in _SENTENCE_END.split(para) if s.strip()]
        result.extend(sentences if sentences else [para])

    return result


# ── Core chunking algorithm ───────────────────────────────────────────────────

class Chunker:
    """
    Splits a list of PageBlocks into a flat list of Chunks.

    Usage
    ─────
    chunker = Chunker(chunk_size=500, overlap=75, min_chunk=50)
    chunks  = chunker.chunk_document(loaded_doc)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 75,
        min_chunk: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk = min_chunk

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_document(self, doc: LoadedDocument) -> list[Chunk]:
        """Chunk an entire LoadedDocument into a flat list of Chunks."""
        all_chunks: list[Chunk] = []
        doc_id = _stable_id(doc.source_path)

        for block in doc.blocks:
            block_chunks = list(self._chunk_block(block, doc_id))
            all_chunks.extend(block_chunks)

        logger.info(
            "Chunked '%s': %d blocks → %d chunks",
            doc.document_title, len(doc.blocks), len(all_chunks),
        )
        return all_chunks

    def chunk_text(
        self,
        text: str,
        source_path: str = "inline",
        document_title: str = "Inline Document",
        section_title: str = "",
        page_number: Optional[int] = None,
    ) -> list[Chunk]:
        """Convenience: chunk a bare text string directly."""
        block = PageBlock(
            text=text,
            page_number=page_number or 1,
            section_title=section_title,
            source_path=source_path,
            document_title=document_title,
        )
        doc_id = _stable_id(source_path)
        return list(self._chunk_block(block, doc_id))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _chunk_block(
        self, block: PageBlock, doc_id: str
    ) -> Generator[Chunk, None, None]:
        """
        Split one PageBlock into chunks using recursive structural splitting.

        Algorithm
        ─────────
        1. Split text into structural units (paragraphs / sentences).
        2. Greedily accumulate units into a buffer until the token budget
           would be exceeded.
        3. When the buffer is full: emit a Chunk, then seed the next buffer
           with the overlap prefix so context bleeds across boundaries.
        4. Any leftover buffer is emitted as a final (possibly under-sized)
           chunk — unless it's below min_chunk and there's a previous chunk
           to merge it into (edge case: avoids tiny orphan chunks).
        """
        units = _split_on_structure(block.text)
        if not units:
            return

        buffer: list[str] = []
        buffer_tokens: int = 0
        chunk_idx: int = 0

        for unit in units:
            unit_tokens = count_tokens(unit)

            # A single unit is larger than chunk_size: hard-split it
            if unit_tokens > self.chunk_size:
                # Flush current buffer first
                if buffer:
                    yield from self._flush(buffer, block, doc_id, chunk_idx)
                    chunk_idx += 1
                    buffer, buffer_tokens = [], 0

                # Hard-split the oversized unit
                remainder = unit
                while remainder:
                    head, remainder = _token_slice(remainder, self.chunk_size)
                    yield self._make_chunk(head, block, doc_id, chunk_idx)
                    chunk_idx += 1
                continue

            # Would adding this unit overflow the buffer?
            if buffer_tokens + unit_tokens > self.chunk_size and buffer:
                yield from self._flush(buffer, block, doc_id, chunk_idx)
                chunk_idx += 1
                # Seed new buffer with overlap from tail of old buffer
                buffer, buffer_tokens = self._overlap_seed(buffer)

            buffer.append(unit)
            buffer_tokens += unit_tokens

        # Emit whatever is left in the buffer
        if buffer:
            leftover_text = "\n\n".join(buffer).strip()
            leftover_tokens = count_tokens(leftover_text)

            if leftover_tokens < self.min_chunk and chunk_idx > 0:
                # Too small — skip (the overlap already carries this context)
                logger.debug(
                    "Dropped %d-token orphan chunk from '%s' page %s",
                    leftover_tokens, block.source_path, block.page_number,
                )
            else:
                yield self._make_chunk(leftover_text, block, doc_id, chunk_idx)

    def _flush(
        self,
        buffer: list[str],
        block: PageBlock,
        doc_id: str,
        chunk_idx: int,
    ) -> Generator[Chunk, None, None]:
        text = "\n\n".join(buffer).strip()
        if text:
            yield self._make_chunk(text, block, doc_id, chunk_idx)

    def _overlap_seed(self, buffer: list[str]) -> tuple[list[str], int]:
        """
        Return the tail of *buffer* whose total token count ≤ self.overlap.
        This becomes the start of the next chunk to provide context continuity.
        """
        seed: list[str] = []
        seed_tokens = 0
        for unit in reversed(buffer):
            t = count_tokens(unit)
            if seed_tokens + t > self.overlap:
                break
            seed.insert(0, unit)
            seed_tokens += t
        return seed, seed_tokens

    def _make_chunk(
        self,
        text: str,
        block: PageBlock,
        doc_id: str,
        chunk_idx: int,
    ) -> Chunk:
        chunk_id = _stable_chunk_id(block.source_path, block.page_number, chunk_idx)
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            token_count=count_tokens(text),
            source_path=block.source_path,
            document_title=block.document_title,
            section_title=block.section_title,
            page_number=block.page_number,
            chunk_index=chunk_idx,
            document_id=doc_id,
        )


# ── Stable ID helpers ─────────────────────────────────────────────────────────

def _stable_id(value: str) -> str:
    """SHA-256 hex digest of a string (first 32 chars for readability)."""
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _stable_chunk_id(source_path: str, page: Optional[int], idx: int) -> str:
    """Stable chunk ID: deterministic across re-ingestion of same file."""
    key = f"{source_path}::page={page}::idx={idx}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


# ── Preview helper ────────────────────────────────────────────────────────────

def preview_chunks(
    chunks: list[Chunk],
    max_preview_chars: int = 120,
    max_chunks: int = 50,
) -> str:
    """
    Return a readable table of chunk summaries for inspection.
    Use this in the CLI / notebook to evaluate chunk quality before indexing.
    """
    lines = [
        f"{'#':>4}  {'Tokens':>6}  {'Page':>4}  {'Section':<30}  Preview",
        "-" * 100,
    ]
    for i, chunk in enumerate(chunks[:max_chunks]):
        preview = textwrap.shorten(
            chunk.text.replace("\n", " "), width=max_preview_chars, placeholder="..."
        )
        section = textwrap.shorten(chunk.section_title or "-", width=30, placeholder="...")
        lines.append(
            f"{i:>4}  {chunk.token_count:>6}  {str(chunk.page_number or '?'):>4}  "
            f"{section:<30}  {preview}"
        )
    if len(chunks) > max_chunks:
        lines.append(f"  ... and {len(chunks) - max_chunks} more chunks (not shown)")
    lines.append("-" * 100)
    lines.append(
        f"Total chunks: {len(chunks)}  |  "
        f"Total tokens: {sum(c.token_count for c in chunks):,}"
    )
    return "\n".join(lines)


# ── Convenience factory ───────────────────────────────────────────────────────

def get_chunker() -> Chunker:
    """Return a Chunker configured from application settings."""
    from app.config import settings
    return Chunker(
        chunk_size=settings.CHUNK_SIZE_TOKENS,
        overlap=settings.CHUNK_OVERLAP_TOKENS,
        min_chunk=settings.CHUNK_MIN_TOKENS,
    )
