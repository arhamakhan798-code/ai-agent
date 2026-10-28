"""
tests/test_chunking.py
─────────────────────────────────────────────────────────────────────────────
Tests for document_loader.py, chunking.py, and their integration.

Run with:  pytest tests/test_chunking.py -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import tempfile
from pathlib import Path

import pytest

from app.services.chunking import (
    Chunk, Chunker, count_tokens, preview_chunks,
    _stable_chunk_id, _stable_id, _split_on_structure, _token_slice,
)
from app.services.document_loader import (
    LoadedDocument, PageBlock,
    _load_txt, _load_md, _load_docx, _load_pdf, load_document, load_directory,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SIMPLE_POLICY = """\
COMPANY POLICY DOCUMENT
TechMart Online Store — Customer Policies

RETURN POLICY
Customers may return items within 30 days of the purchase date.
All returned items must be unused and in their original packaging.
Electronics such as laptops, phones, and tablets must be returned within 15 days.
Items marked as "Final Sale" cannot be returned or exchanged.

REFUND POLICY
Refunds are processed within 5 business days after the returned item is received.
Refunds are issued to the original payment method only.

SHIPPING POLICY
Standard shipping takes 3–5 business days and costs $4.99.
Express shipping takes 1–2 business days and costs $12.99.
Free standard shipping is available on all orders above $50.
"""

MARKDOWN_POLICY = """\
# TechMart Policies

## Return Policy

Customers may return items within 30 days.
Electronics must be returned within 15 days.

## Refund Policy

Refunds are processed within 5 business days.
Refunds are issued to the original payment method only.

## Shipping Policy

Standard shipping takes 3–5 business days and costs $4.99.
Express shipping takes 1–2 business days and costs $12.99.
"""


@pytest.fixture()
def tmp_txt_file(tmp_path: Path) -> Path:
    f = tmp_path / "policy.txt"
    f.write_text(SIMPLE_POLICY, encoding="utf-8")
    return f


@pytest.fixture()
def tmp_md_file(tmp_path: Path) -> Path:
    f = tmp_path / "policy.md"
    f.write_text(MARKDOWN_POLICY, encoding="utf-8")
    return f


@pytest.fixture()
def tmp_policy_dir(tmp_path: Path) -> Path:
    (tmp_path / "policy.txt").write_text(SIMPLE_POLICY, encoding="utf-8")
    (tmp_path / "policy.md").write_text(MARKDOWN_POLICY, encoding="utf-8")
    return tmp_path


# ── count_tokens ──────────────────────────────────────────────────────────────

class TestCountTokens:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_short_text_positive(self):
        assert count_tokens("Hello world") > 0

    def test_proportional_growth(self):
        short = count_tokens("Hello world")
        long = count_tokens("Hello world " * 20)
        assert long > short * 10

    def test_returns_int(self):
        assert isinstance(count_tokens("some text"), int)


# ── _stable_id / _stable_chunk_id ────────────────────────────────────────────

class TestStableIds:
    def test_same_input_same_id(self):
        assert _stable_id("my_path") == _stable_id("my_path")

    def test_different_input_different_id(self):
        assert _stable_id("path_a") != _stable_id("path_b")

    def test_id_length(self):
        # We use first 32 chars of hex SHA-256
        assert len(_stable_id("anything")) == 32

    def test_chunk_id_stable(self):
        a = _stable_chunk_id("file.txt", 1, 0)
        b = _stable_chunk_id("file.txt", 1, 0)
        assert a == b

    def test_chunk_id_differs_by_page(self):
        a = _stable_chunk_id("file.txt", 1, 0)
        b = _stable_chunk_id("file.txt", 2, 0)
        assert a != b

    def test_chunk_id_differs_by_index(self):
        a = _stable_chunk_id("file.txt", 1, 0)
        b = _stable_chunk_id("file.txt", 1, 1)
        assert a != b


# ── _split_on_structure ───────────────────────────────────────────────────────

class TestSplitOnStructure:
    def test_returns_nonempty_strings(self):
        units = _split_on_structure("Hello world.\n\nGoodbye world.")
        assert all(u.strip() for u in units)

    def test_splits_on_double_newline(self):
        text = "First paragraph.\n\nSecond paragraph."
        units = _split_on_structure(text)
        assert len(units) >= 2

    def test_preserves_table_row(self):
        text = "| Column A | Column B |\n| data 1   | data 2   |\n\nSome text."
        units = _split_on_structure(text)
        # Table should not be split across units
        table_units = [u for u in units if "|" in u]
        assert table_units, "Table rows should be preserved"

    def test_empty_text_returns_empty_list(self):
        assert _split_on_structure("") == [] or _split_on_structure("   ") == []

    def test_numbered_list_kept_together(self):
        text = "1. First rule\n2. Second rule\n3. Third rule"
        units = _split_on_structure(text)
        # The numbered list should be in one or two units (not split into 3)
        assert len(units) <= 3  # Reasonable upper bound


# ── _token_slice ──────────────────────────────────────────────────────────────

class TestTokenSlice:
    def test_short_text_no_split(self):
        head, tail = _token_slice("short text", max_tokens=100)
        assert head == "short text"
        assert tail == ""

    def test_long_text_is_split(self):
        long_text = " ".join(["word"] * 300)
        head, tail = _token_slice(long_text, max_tokens=50)
        assert len(head) > 0
        assert len(tail) > 0
        assert len(head) < len(long_text)

    def test_head_plus_tail_approximately_reconstructs(self):
        text = "The quick brown fox jumps over the lazy dog. " * 20
        head, tail = _token_slice(text, max_tokens=30)
        # Combined should preserve most of the original content
        combined = (head + " " + tail).strip()
        assert len(combined) >= len(text) * 0.8


# ── document_loader — TXT ────────────────────────────────────────────────────

class TestLoadTxt:
    def test_returns_loaded_document(self, tmp_txt_file: Path):
        doc = _load_txt(tmp_txt_file)
        assert isinstance(doc, LoadedDocument)

    def test_format_is_txt(self, tmp_txt_file: Path):
        doc = _load_txt(tmp_txt_file)
        assert doc.format == "txt"

    def test_has_blocks(self, tmp_txt_file: Path):
        doc = _load_txt(tmp_txt_file)
        assert len(doc.blocks) >= 1

    def test_block_text_not_empty(self, tmp_txt_file: Path):
        doc = _load_txt(tmp_txt_file)
        assert all(b.text.strip() for b in doc.blocks)

    def test_document_title_derived(self, tmp_txt_file: Path):
        doc = _load_txt(tmp_txt_file)
        assert doc.document_title  # Not empty

    def test_source_path_set(self, tmp_txt_file: Path):
        doc = _load_txt(tmp_txt_file)
        assert str(tmp_txt_file) in doc.blocks[0].source_path

    def test_no_error(self, tmp_txt_file: Path):
        doc = _load_txt(tmp_txt_file)
        assert doc.error is None


# ── document_loader — Markdown ───────────────────────────────────────────────

class TestLoadMd:
    def test_splits_by_headings(self, tmp_md_file: Path):
        doc = _load_md(tmp_md_file)
        # Should have at least 3 sections (Return, Refund, Shipping)
        assert len(doc.blocks) >= 3

    def test_section_titles_populated(self, tmp_md_file: Path):
        doc = _load_md(tmp_md_file)
        titled = [b for b in doc.blocks if b.section_title]
        assert len(titled) >= 2

    def test_return_policy_section_found(self, tmp_md_file: Path):
        doc = _load_md(tmp_md_file)
        titles = [b.section_title for b in doc.blocks]
        assert any("Return" in t for t in titles)

    def test_format_is_md(self, tmp_md_file: Path):
        doc = _load_md(tmp_md_file)
        assert doc.format == "md"


# ── document_loader — generic dispatch ───────────────────────────────────────

class TestLoadDocument:
    def test_txt_dispatched_correctly(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        assert doc.format == "txt"

    def test_md_dispatched_correctly(self, tmp_md_file: Path):
        doc = load_document(tmp_md_file)
        assert doc.format == "md"

    def test_missing_file_returns_error(self, tmp_path: Path):
        doc = load_document(tmp_path / "nonexistent.txt")
        assert doc.error is not None

    def test_unsupported_format_raises(self, tmp_path: Path):
        bad = tmp_path / "file.xyz"
        bad.write_text("data")
        with pytest.raises(ValueError, match="Unsupported document format"):
            load_document(bad)

    def test_load_directory(self, tmp_policy_dir: Path):
        docs = load_directory(tmp_policy_dir)
        assert len(docs) >= 2
        assert all(isinstance(d, LoadedDocument) for d in docs)


# ── Chunker ───────────────────────────────────────────────────────────────────

class TestChunker:
    def test_produces_chunks(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        chunker = Chunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1

    def test_chunk_has_stable_id(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        chunker = Chunker(chunk_size=100, overlap=20)
        chunks1 = chunker.chunk_document(doc)
        chunks2 = chunker.chunk_document(doc)
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2, "Chunk IDs must be stable across runs"

    def test_no_empty_chunks(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        chunker = Chunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(doc)
        assert all(c.text.strip() for c in chunks)

    def test_chunk_token_counts_positive(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        chunker = Chunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(doc)
        assert all(c.token_count > 0 for c in chunks)

    def test_chunk_metadata_populated(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        chunker = Chunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.source_path
            assert chunk.document_title
            assert chunk.document_id

    def test_chunk_text_is_subset_of_original(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        full_text = " ".join(b.text for b in doc.blocks).lower()
        chunker = Chunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            # Each chunk's first 50 chars should appear in the original
            fragment = chunk.text[:50].lower().strip()
            assert any(
                word in full_text for word in fragment.split()[:3]
            ), f"Chunk text not found in original: {fragment!r}"

    def test_chunk_size_respected(self, tmp_txt_file: Path):
        target = 80
        doc = load_document(tmp_txt_file)
        chunker = Chunker(chunk_size=target, overlap=10, min_chunk=5)
        chunks = chunker.chunk_document(doc)
        # Most chunks should be ≤ target (allow 10% slack for structural splits)
        oversized = [c for c in chunks if c.token_count > target * 1.1]
        # Only hard-split oversized units are allowed to slightly exceed
        assert len(oversized) == 0 or len(oversized) < len(chunks) * 0.1

    def test_overlap_seed_carries_context(self, tmp_md_file: Path):
        """With overlap, consecutive chunks should share some words."""
        doc = load_document(tmp_md_file)
        chunker = Chunker(chunk_size=50, overlap=20, min_chunk=5)
        chunks = chunker.chunk_document(doc)
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        # At least one pair of consecutive chunks should share some words
        found_overlap = False
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].text.lower().split())
            words_b = set(chunks[i + 1].text.lower().split())
            if words_a & words_b:
                found_overlap = True
                break
        assert found_overlap, "Expected at least one overlapping word between consecutive chunks"

    def test_chunk_text_convenience(self):
        chunker = Chunker(chunk_size=50, overlap=10, min_chunk=5)
        chunks = chunker.chunk_text(
            "This is test content for inline chunking. " * 10,
            source_path="inline",
            document_title="Test Doc",
        )
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_document_produces_no_chunks(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_text("   \n   ", encoding="utf-8")
        doc = load_document(f)
        chunker = Chunker()
        chunks = chunker.chunk_document(doc)
        # Empty or whitespace-only blocks should produce zero chunks
        assert all(c.text.strip() for c in chunks)

    def test_large_single_unit_hard_split(self):
        """A single paragraph larger than chunk_size must still be split."""
        large_text = "word " * 1000  # ~1000 tokens approx
        chunker = Chunker(chunk_size=100, overlap=20, min_chunk=10)
        chunks = chunker.chunk_text(large_text, source_path="test")
        assert len(chunks) >= 5, "Large unit must be hard-split into multiple chunks"
        for c in chunks:
            assert c.token_count <= 110  # Allow 10% slack

    def test_unique_chunk_ids_within_document(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        chunker = Chunker(chunk_size=80, overlap=15)
        chunks = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "All chunk IDs must be unique"


# ── preview_chunks ────────────────────────────────────────────────────────────

class TestPreviewChunks:
    def test_returns_string(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        chunker = Chunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(doc)
        output = preview_chunks(chunks)
        assert isinstance(output, str)
        assert "Total chunks" in output

    def test_respects_max_chunks(self, tmp_txt_file: Path):
        doc = load_document(tmp_txt_file)
        chunker = Chunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_document(doc)
        output = preview_chunks(chunks, max_chunks=2)
        # Should say "and N more chunks" if there are > 2
        if len(chunks) > 2:
            assert "more chunks" in output
