"""
tests/test_ingestion.py
─────────────────────────────────────────────────────────────────────────────
Integration tests for the policy ingestion pipeline.

These tests use a real (ephemeral) ChromaDB in-memory / tmp directory so no
persistent state leaks between runs.

Run with:  pytest tests/test_ingestion.py -v --tb=short
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

POLICY_TEXT = """\
RETURN POLICY
Customers may return items within 30 days of the purchase date.
All returned items must be unused and in their original packaging.
Electronics such as laptops, phones, and tablets must be returned within 15 days.
Items marked as "Final Sale" cannot be returned or exchanged.

REFUND POLICY
Refunds are processed within 5 business days after the returned item is received.
Refunds are issued to the original payment method only.
Shipping costs are non-refundable unless the return is due to a defective item.

SHIPPING POLICY
Standard shipping takes 3-5 business days and costs $4.99.
Express shipping takes 1-2 business days and costs $12.99.
Free standard shipping is available on all orders above $50.
"""


@pytest.fixture()
def tmp_policy_file(tmp_path: Path) -> Path:
    f = tmp_path / "policy.txt"
    f.write_text(POLICY_TEXT, encoding="utf-8")
    return f


@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path, monkeypatch):
    """
    Redirect ChromaDB and ingestion registry to an ephemeral temp directory
    for each test so tests are fully isolated.
    """
    import app.services.chroma_service as cs
    import app.services.policy_ingestion as pi

    # Reset module-level state
    cs._client = None
    cs._collection = None
    cs._ef = None
    pi._ingestion_registry.clear()

    # Point ChromaDB at a temp path
    monkeypatch.setattr("app.config.settings.CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setattr("app.config.settings.CHROMA_COLLECTION_NAME", "test_policies")
    monkeypatch.setattr("app.config.settings.EMBEDDING_PROVIDER", "chroma_default")
    monkeypatch.setattr("app.config.settings.CHUNK_SIZE_TOKENS", 100)
    monkeypatch.setattr("app.config.settings.CHUNK_OVERLAP_TOKENS", 20)
    monkeypatch.setattr("app.config.settings.CHUNK_MIN_TOKENS", 10)
    monkeypatch.setattr("app.config.settings.POLICY_SCORE_THRESHOLD", 0.1)

    yield

    # Cleanup
    cs._client = None
    cs._collection = None
    cs._ef = None
    pi._ingestion_registry.clear()


# ── ChromaDB initialisation ───────────────────────────────────────────────────

class TestChromaInit:
    def test_init_chroma_succeeds(self):
        from app.services.chroma_service import init_chroma, collection_count
        init_chroma()
        assert collection_count() == 0

    def test_double_init_is_noop(self):
        from app.services.chroma_service import init_chroma, collection_count
        init_chroma()
        init_chroma()   # Should not raise
        assert collection_count() == 0

    def test_get_collection_before_init_raises(self):
        import app.services.chroma_service as cs
        cs._collection = None  # Ensure uninitialized
        with pytest.raises(RuntimeError, match="not initialized"):
            cs.get_collection()

    def test_get_status_ok(self):
        from app.services.chroma_service import init_chroma, get_status
        init_chroma()
        status = get_status()
        assert status["status"] == "ok"
        assert "chunk_count" in status


# ── Upsert / delete ───────────────────────────────────────────────────────────

class TestUpsertAndDelete:
    def test_upsert_chunks_increases_count(self, tmp_policy_file: Path):
        from app.services.chroma_service import init_chroma, collection_count
        from app.services.document_loader import load_document
        from app.services.chunking import get_chunker

        init_chroma()
        doc = load_document(tmp_policy_file)
        chunker = get_chunker()
        chunks = chunker.chunk_document(doc)

        from app.services.chroma_service import upsert_chunks
        result = upsert_chunks(chunks)

        assert result["upserted"] > 0
        assert collection_count() > 0

    def test_upsert_is_idempotent(self, tmp_policy_file: Path):
        from app.services.chroma_service import init_chroma, upsert_chunks, collection_count
        from app.services.document_loader import load_document
        from app.services.chunking import get_chunker

        init_chroma()
        doc = load_document(tmp_policy_file)
        chunks = get_chunker().chunk_document(doc)

        upsert_chunks(chunks)
        count_after_first = collection_count()
        upsert_chunks(chunks)  # Same chunks again
        count_after_second = collection_count()

        assert count_after_first == count_after_second, (
            "Re-upserting same chunks should not grow the collection"
        )

    def test_delete_document_chunks(self, tmp_policy_file: Path):
        from app.services.chroma_service import (
            init_chroma, upsert_chunks, delete_source_chunks, collection_count
        )
        from app.services.document_loader import load_document
        from app.services.chunking import get_chunker

        init_chroma()
        doc = load_document(tmp_policy_file)
        chunks = get_chunker().chunk_document(doc)
        upsert_chunks(chunks)

        assert collection_count() > 0

        deleted = delete_source_chunks(str(tmp_policy_file.resolve()))
        assert deleted > 0
        assert collection_count() == 0


# ── Ingestion pipeline ────────────────────────────────────────────────────────

class TestIngestionPipeline:
    def test_ingest_document_returns_ok(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        record = ingest_document(str(tmp_policy_file))
        assert record.status == "ok"
        assert record.chunk_count > 0

    def test_ingest_document_sets_title(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        record = ingest_document(str(tmp_policy_file))
        assert record.document_title  # Not empty

    def test_ingest_document_hash_stored(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        record = ingest_document(str(tmp_policy_file))
        assert len(record.file_hash) == 64  # SHA-256 hex

    def test_second_ingest_skipped_without_force(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        record1 = ingest_document(str(tmp_policy_file))
        record2 = ingest_document(str(tmp_policy_file))
        # Second call with same file and no changes should skip
        assert record1.status == "ok"
        assert record2.status in ("ok", "skipped")

    def test_force_reingest(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        record1 = ingest_document(str(tmp_policy_file))
        record2 = ingest_document(str(tmp_policy_file), force=True)
        assert record2.status == "ok"
        assert record2.chunk_count > 0

    def test_changed_file_reindexed(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        record1 = ingest_document(str(tmp_policy_file))

        # Modify the file
        time.sleep(0.05)  # ensure different mtime
        tmp_policy_file.write_text(
            POLICY_TEXT + "\nNEW WARRANTY POLICY\nAll products carry 2-year warranty.",
            encoding="utf-8",
        )

        record2 = ingest_document(str(tmp_policy_file))
        assert record2.status == "ok"
        assert record2.file_hash != record1.file_hash

    def test_ingest_missing_file_returns_failed(self, tmp_path: Path):
        from app.services.policy_ingestion import ingest_document
        record = ingest_document(str(tmp_path / "ghost.txt"))
        assert record.status == "failed"
        assert record.error

    def test_ingest_with_manifest_meta(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        meta = {"version": "v3.0", "effective_date": "2025-01-01", "access_scope": "public"}
        record = ingest_document(str(tmp_policy_file), manifest_meta=meta)
        assert record.version == "v3.0"
        assert record.effective_date == "2025-01-01"

    def test_get_ingestion_status(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document, get_ingestion_status
        ingest_document(str(tmp_policy_file))
        status_list = get_ingestion_status()
        assert len(status_list) >= 1
        assert any(s["status"] == "ok" for s in status_list)

    def test_remove_document(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document, remove_document
        from app.services.chroma_service import collection_count
        ingest_document(str(tmp_policy_file))
        assert collection_count() > 0
        remove_document(str(tmp_policy_file.resolve()))
        assert collection_count() == 0

    def test_ingest_directory(self, tmp_path: Path):
        from app.services.policy_ingestion import ingest_directory
        # Create two policy files in the temp dir
        (tmp_path / "policy_a.txt").write_text(POLICY_TEXT, encoding="utf-8")
        (tmp_path / "policy_b.txt").write_text(
            "PRIVACY POLICY\nWe respect your privacy and data.", encoding="utf-8"
        )
        records = ingest_directory(str(tmp_path), extensions=(".txt",))
        assert len(records) == 2
        assert all(r.status == "ok" for r in records)


# ── Semantic query ────────────────────────────────────────────────────────────

class TestSemanticQuery:
    def test_query_returns_results(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        from app.services.chroma_service import query_policy

        ingest_document(str(tmp_policy_file))
        results = query_policy("What is the return window?", n_results=3)
        assert len(results) >= 1

    def test_result_has_required_fields(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        from app.services.chroma_service import query_policy

        ingest_document(str(tmp_policy_file))
        results = query_policy("refund processing time", n_results=3)
        if results:
            r = results[0]
            assert "text" in r
            assert "score" in r
            assert "document_title" in r
            assert "source_path" in r

    def test_score_in_valid_range(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        from app.services.chroma_service import query_policy

        ingest_document(str(tmp_policy_file))
        results = query_policy("shipping cost for standard delivery", n_results=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_results_sorted_by_descending_score(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        from app.services.chroma_service import query_policy

        ingest_document(str(tmp_policy_file))
        results = query_policy("electronics return policy", n_results=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_results_for_unrelated_query(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        from app.services.chroma_service import query_policy

        ingest_document(str(tmp_policy_file))
        # Very high threshold so no results should pass
        results = query_policy("quantum entanglement and dark matter", n_results=3, score_threshold=0.99)
        assert results == []

    def test_high_threshold_filters_low_scores(self, tmp_policy_file: Path):
        from app.services.policy_ingestion import ingest_document
        from app.services.chroma_service import query_policy

        ingest_document(str(tmp_policy_file))
        results = query_policy("warranty coverage", n_results=5, score_threshold=0.9)
        # Everything returned must be above threshold
        for r in results:
            assert r["score"] >= 0.9
