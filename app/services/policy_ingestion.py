"""
app/services/policy_ingestion.py
─────────────────────────────────────────────────────────────────────────────
End-to-end policy document ingestion pipeline.

One call to `ingest_document()` handles:
    1. Load → PageBlocks  (document_loader)
    2. Chunk → Chunks     (chunking)
    3. Embed & Upsert     (chroma_service)
    4. Track status       (in-memory registry, queryable via get_ingestion_status)

Incremental updates
───────────────────
Documents are identified by the SHA-256 of their absolute source_path.
If a document is re-ingested with changed content:
    • Old chunks (same document_id) are deleted first.
    • New chunks are upserted.
This ensures stale content never survives a re-index.

File-hash check
───────────────
`ingest_directory()` computes a SHA-256 of each file's content.
If the hash matches the last ingested hash stored in _ingestion_registry,
the file is skipped (no-op).  This avoids re-embedding unchanged documents.

Manifest support
────────────────
An optional JSON manifest file can supply per-document metadata:
    {
      "policies/return_policy.pdf": {
        "version": "v2.1",
        "effective_date": "2024-01-01",
        "access_scope": "public"
      }
    }
Pass the manifest path to `ingest_directory()` or `ingest_document()`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Ingestion registry (in-memory) ────────────────────────────────────────────

@dataclass
class IngestionRecord:
    source_path: str
    document_title: str
    file_hash: str
    chunk_count: int
    status: str          # "ok" | "failed" | "skipped"
    error: Optional[str] = None
    ingested_at: float = field(default_factory=time.time)
    version: str = ""
    effective_date: str = ""
    access_scope: str = "public"


# module-level registry: source_path → IngestionRecord
_ingestion_registry: dict[str, IngestionRecord] = {}


def get_ingestion_status() -> list[dict[str, Any]]:
    """Return the current in-memory ingestion registry as a list of dicts."""
    return [
        {
            "source_path": r.source_path,
            "document_title": r.document_title,
            "file_hash": r.file_hash,
            "chunk_count": r.chunk_count,
            "status": r.status,
            "error": r.error,
            "ingested_at": r.ingested_at,
            "version": r.version,
            "effective_date": r.effective_date,
            "access_scope": r.access_scope,
        }
        for r in _ingestion_registry.values()
    ]


# ── File hash ─────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """SHA-256 of file bytes — used to detect unchanged files."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ── Core ingestion functions ───────────────────────────────────────────────────

def ingest_document(
    path: str | Path,
    manifest_meta: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> IngestionRecord:
    """
    Ingest a single document into ChromaDB.

    Parameters
    ──────────
    path          Absolute or relative path to the document file.
    manifest_meta Dict with optional keys: version, effective_date, access_scope.
    force         If True, re-ingest even if the file hash hasn't changed.

    Returns the IngestionRecord for this document.
    """
    from app.services.document_loader import load_document
    from app.services.chunking import get_chunker, Chunk
    from app.services.chroma_service import (
        upsert_chunks, delete_document_chunks, init_chroma
    )
    from app.services.chunking import _stable_id

    init_chroma()

    p = Path(path).resolve()
    source_path_str = str(p)
    meta = manifest_meta or {}

    # ── File hash check ───────────────────────────────────────────────────────
    current_hash = _file_hash(p) if p.exists() else ""

    existing = _ingestion_registry.get(source_path_str)
    if (
        not force
        and existing
        and existing.file_hash == current_hash
        and existing.status in ("ok", "skipped")
    ):
        logger.info("Skipping unchanged document: '%s'", p.name)
        return IngestionRecord(
            source_path=existing.source_path,
            document_title=existing.document_title,
            file_hash=existing.file_hash,
            chunk_count=existing.chunk_count,
            status="skipped",
            version=existing.version,
            effective_date=existing.effective_date,
            access_scope=existing.access_scope,
            ingested_at=existing.ingested_at,
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    doc = load_document(p)

    if doc.error:
        record = IngestionRecord(
            source_path=source_path_str,
            document_title=doc.document_title,
            file_hash=current_hash,
            chunk_count=0,
            status="failed",
            error=doc.error,
        )
        _ingestion_registry[source_path_str] = record
        logger.error("Ingestion failed for '%s': %s", p.name, doc.error)
        return record

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunker = get_chunker()
    chunks = chunker.chunk_document(doc)

    # Attach manifest metadata to every chunk
    version = meta.get("version", "")
    effective_date = meta.get("effective_date", "")
    access_scope = meta.get("access_scope", "public")

    for chunk in chunks:
        chunk.version = version
        chunk.effective_date = effective_date
        chunk.access_scope = access_scope

    # ── Delete old chunks before upserting new ones ───────────────────────────
    doc_id = _stable_id(source_path_str)
    deleted = delete_document_chunks(doc_id)
    if deleted:
        logger.info(
            "Removed %d stale chunks for '%s' before re-indexing",
            deleted, p.name,
        )

    # ── Upsert ────────────────────────────────────────────────────────────────
    upsert_result = upsert_chunks(chunks)

    record = IngestionRecord(
        source_path=source_path_str,
        document_title=doc.document_title,
        file_hash=current_hash,
        chunk_count=upsert_result["upserted"],
        status="ok",
        version=version,
        effective_date=effective_date,
        access_scope=access_scope,
    )
    _ingestion_registry[source_path_str] = record

    logger.info(
        "Ingested '%s': %d chunks upserted (doc_id=%s)",
        p.name, upsert_result["upserted"], doc_id,
    )
    return record


def remove_document(source_path: str) -> int:
    """
    Remove all chunks for a document from ChromaDB and the registry.
    Returns the number of chunks deleted.
    """
    from app.services.chroma_service import delete_source_chunks

    deleted = delete_source_chunks(source_path)
    _ingestion_registry.pop(source_path, None)
    logger.info("Removed document '%s' (%d chunks deleted)", source_path, deleted)
    return deleted


def ingest_directory(
    directory: str | Path,
    extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx"),
    recursive: bool = True,
    manifest_path: Optional[str | Path] = None,
    force: bool = False,
) -> list[IngestionRecord]:
    """
    Batch-ingest all supported documents in a directory.

    Parameters
    ──────────
    directory     Root directory to scan.
    extensions    File extensions to process.
    recursive     Whether to descend into subdirectories.
    manifest_path Path to a JSON manifest with per-file metadata.
    force         Re-ingest all files even if unchanged.

    Returns a list of IngestionRecord for every file processed.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    # Load manifest if provided
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            logger.info("Loaded manifest with %d entries", len(manifest))
        except Exception as exc:
            logger.warning("Could not load manifest '%s': %s", manifest_path, exc)

    pattern = "**/*" if recursive else "*"
    records: list[IngestionRecord] = []

    for ext in extensions:
        for file_path in root.glob(f"{pattern}{ext}"):
            # Look for metadata in manifest keyed by relative or absolute path
            rel_key = str(file_path.relative_to(root))
            meta = manifest.get(rel_key) or manifest.get(str(file_path)) or {}

            record = ingest_document(file_path, manifest_meta=meta, force=force)
            records.append(record)

    ok = sum(1 for r in records if r.status == "ok")
    skipped = sum(1 for r in records if r.status == "skipped")
    failed = sum(1 for r in records if r.status == "failed")

    logger.info(
        "Batch ingestion complete: %d ok, %d skipped, %d failed (total %d files)",
        ok, skipped, failed, len(records),
    )
    return records
