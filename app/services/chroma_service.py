"""
app/services/chroma_service.py
─────────────────────────────────────────────────────────────────────────────
ChromaDB vector store wrapper — handles:

    • Initialization of the persistent client and collection.
    • Incremental upsert (add-or-replace by stable chunk_id).
    • Deletion of chunks belonging to a removed or superseded document.
    • Similarity search with metadata filtering and score thresholding.
    • Deduplication of overlapping results.
    • Ingestion status tracking (in-memory; persisted to sync_states table
      separately by the ingestion pipeline).

ChromaDB version compatibility
───────────────────────────────
Tested against chromadb >= 0.5.x.  The API used here is stable across
0.4.x – 1.x.  The only version-specific call is `collection.upsert()` with
`ids`, `documents`, and `metadatas` which has been stable since 0.3.x.

Thread safety
─────────────
ChromaDB's PersistentClient is thread-safe for reads.  Writes are serialised
by a module-level threading.Lock so concurrent ingestion calls don't corrupt
the collection.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.services.chunking import Chunk

logger = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_ef = None          # embedding function
_lock = threading.Lock()


# ── Initialization ────────────────────────────────────────────────────────────

def init_chroma() -> None:
    """
    Initialize the ChromaDB persistent client and policy collection.
    Call once at application startup (FastAPI lifespan / script entry).
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _client, _collection, _ef

    if _collection is not None:
        return   # Already initialized

    from app.config import settings
    from app.services.embeddings import get_embedding_function

    persist_dir = settings.CHROMA_PERSIST_DIRECTORY
    collection_name = settings.CHROMA_COLLECTION_NAME

    logger.info("Initializing ChromaDB at '%s'", persist_dir)
    _client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    _ef = get_embedding_function()

    _collection = _client.get_or_create_collection(
        name=collection_name,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )

    count = _collection.count()
    logger.info(
        "ChromaDB ready — collection '%s' contains %d chunks",
        collection_name, count,
    )


def get_collection() -> chromadb.Collection:
    """Return the initialized collection, raising if not yet initialized."""
    if _collection is None:
        raise RuntimeError(
            "ChromaDB not initialized. Call init_chroma() first."
        )
    return _collection


# ── Ingestion ─────────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[Chunk]) -> dict[str, int]:
    """
    Upsert a list of Chunk objects into ChromaDB.

    Uses chroma `upsert` so re-ingesting the same chunk (same chunk_id) is
    idempotent — it updates the text and metadata without creating duplicates.

    Returns a dict: {"upserted": N, "skipped": M}
    """
    if not chunks:
        return {"upserted": 0, "skipped": 0}

    collection = get_collection()
    ids, texts, metadatas = [], [], []

    for chunk in chunks:
        if not chunk.text.strip():
            continue
        ids.append(chunk.chunk_id)
        texts.append(chunk.text)
        metadatas.append(_chunk_to_metadata(chunk))

    if not ids:
        return {"upserted": 0, "skipped": len(chunks)}

    with _lock:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    logger.info("Upserted %d chunks into ChromaDB", len(ids))
    return {"upserted": len(ids), "skipped": len(chunks) - len(ids)}


def delete_document_chunks(document_id: str) -> int:
    """
    Delete all chunks belonging to a document (identified by document_id,
    which is the SHA-256 of the source_path).

    Returns the number of chunks deleted.
    """
    collection = get_collection()

    # Query for all chunks with this document_id
    results = collection.get(
        where={"document_id": {"$eq": document_id}},
        include=[],   # We only need IDs
    )
    chunk_ids = results.get("ids", [])

    if not chunk_ids:
        return 0

    with _lock:
        collection.delete(ids=chunk_ids)

    logger.info(
        "Deleted %d chunks for document_id='%s'", len(chunk_ids), document_id
    )
    return len(chunk_ids)


def delete_source_chunks(source_path: str) -> int:
    """
    Delete all chunks from a specific source file path.
    Convenience wrapper around delete_document_chunks.
    """
    from app.services.chunking import _stable_id
    doc_id = _stable_id(source_path)
    return delete_document_chunks(doc_id)


def collection_count() -> int:
    """Return total number of chunks currently in the collection."""
    return get_collection().count()


# ── Retrieval ─────────────────────────────────────────────────────────────────

def query_policy(
    query_text: str,
    n_results: int = 5,
    score_threshold: float = 0.35,
    metadata_filter: Optional[dict[str, Any]] = None,
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """
    Semantic search over policy chunks.

    Returns a list of result dicts sorted by relevance, each containing:
        text, score, chunk_id, document_title, section_title,
        source_path, page_number, document_id, access_scope
    """
    collection = get_collection()

    query_kwargs: dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": min(n_results * 2, collection.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if metadata_filter:
        query_kwargs["where"] = metadata_filter

    try:
        raw = collection.query(**query_kwargs)
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    seen_texts: set[str] = set()

    docs_list = raw.get("documents") or [[]]
    meta_list = raw.get("metadatas") or [[]]
    dist_list = raw.get("distances") or [[]]

    for doc, meta, dist in zip(docs_list[0], meta_list[0], dist_list[0]):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score in [0, 1]
        score = 1.0 - (dist / 2.0)

        if score < score_threshold:
            continue

        if deduplicate:
            # Normalise for dedup comparison (strip whitespace, lowercase)
            norm = " ".join(doc.lower().split())[:200]
            if norm in seen_texts:
                continue
            seen_texts.add(norm)

        results.append({
            "text": doc,
            "score": round(score, 4),
            "chunk_id": meta.get("chunk_id", ""),
            "document_title": meta.get("document_title", ""),
            "section_title": meta.get("section_title", ""),
            "source_path": meta.get("source_path", ""),
            "page_number": meta.get("page_number"),
            "document_id": meta.get("document_id", ""),
            "access_scope": meta.get("access_scope", "public"),
            "version": meta.get("version", ""),
            "effective_date": meta.get("effective_date", ""),
        })

    # Sort by descending score and cap at n_results
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:n_results]


# ── Metadata helpers ──────────────────────────────────────────────────────────

def _chunk_to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Convert a Chunk to a flat dict of scalar metadata for ChromaDB."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_title": chunk.document_title or "",
        "section_title": chunk.section_title or "",
        "source_path": chunk.source_path or "",
        "page_number": chunk.page_number if chunk.page_number is not None else 0,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "version": chunk.version or "",
        "effective_date": chunk.effective_date or "",
        "access_scope": chunk.access_scope or "public",
    }


# ── Status ────────────────────────────────────────────────────────────────────

def get_status() -> dict[str, Any]:
    """Return a summary of ChromaDB state for health/admin endpoints."""
    try:
        col = get_collection()
        count = col.count()
        from app.config import settings
        persist_dir = getattr(getattr(_client, "_settings", None), "persist_directory", settings.CHROMA_PERSIST_DIRECTORY)
        return {
            "status": "ok",
            "collection": col.name,
            "chunk_count": count,
            "persist_directory": str(persist_dir or settings.CHROMA_PERSIST_DIRECTORY),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
