"""
scripts/ingest_policies.py
─────────────────────────────────────────────────────────────────────────────
CLI for ingesting policy documents into ChromaDB.

Usage examples
──────────────
# Ingest all documents in ./data/policies (default)
    python -m scripts.ingest_policies

# Ingest a specific file
    python -m scripts.ingest_policies --file data/policies/return_policy.txt

# Force re-ingest everything (ignore hash cache)
    python -m scripts.ingest_policies --force

# Ingest with a metadata manifest
    python -m scripts.ingest_policies --manifest data/policies/manifest.json

# Preview chunks without indexing
    python -m scripts.ingest_policies --preview --file data/policies/return_policy.txt

# Remove a document from the index
    python -m scripts.ingest_policies --remove data/policies/old_policy.pdf

Run from the project root (ai agent/) so that the app package is on sys.path.
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

# Make sure the project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_policies")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest policy documents into ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Ingest a single document file",
    )
    group.add_argument(
        "--dir", "-d",
        metavar="DIR",
        default="./data/policies",
        help="Directory to batch-ingest (default: ./data/policies)",
    )
    group.add_argument(
        "--remove", "-r",
        metavar="PATH",
        help="Remove a document's chunks from ChromaDB",
    )
    p.add_argument(
        "--manifest", "-m",
        metavar="JSON",
        help="Path to JSON manifest with per-file metadata",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if file hash unchanged",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Print chunk preview without indexing",
    )
    p.add_argument(
        "--extensions",
        nargs="+",
        default=[".txt", ".md", ".pdf", ".docx"],
        metavar="EXT",
        help="File extensions to process (default: .txt .md .pdf .docx)",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print current ChromaDB status and exit",
    )
    return p


def _print_status() -> None:
    from app.services.chroma_service import init_chroma, get_status
    init_chroma()
    status = get_status()
    print("\n-- ChromaDB Status -----------------------------")
    for k, v in status.items():
        print(f"  {k:<25} {v}")
    print()


def _run_preview(file_path: str) -> None:
    from app.services.document_loader import load_document
    from app.services.chunking import get_chunker, preview_chunks

    doc = load_document(file_path)
    if doc.error:
        logger.error("Failed to load '%s': %s", file_path, doc.error)
        sys.exit(1)

    chunker = get_chunker()
    chunks = chunker.chunk_document(doc)

    print(f"\n-- Preview: '{doc.document_title}' ({len(chunks)} chunks) --")
    print(preview_chunks(chunks))


def _run_ingest_file(file_path: str, manifest: str | None, force: bool) -> None:
    from app.services.policy_ingestion import ingest_document

    record = ingest_document(file_path, force=force)
    _print_record(record)


def _run_ingest_dir(
    directory: str,
    extensions: list[str],
    manifest: str | None,
    force: bool,
) -> None:
    from app.services.policy_ingestion import ingest_directory

    records = ingest_directory(
        directory,
        extensions=tuple(extensions),
        manifest_path=manifest,
        force=force,
    )
    print(f"\n-- Ingestion Summary ({len(records)} files) -----------------")
    for r in records:
        _print_record(r)


def _run_remove(source_path: str) -> None:
    from app.services.chroma_service import init_chroma
    from app.services.policy_ingestion import remove_document
    init_chroma()
    deleted = remove_document(source_path)
    print(f"\nRemoved {deleted} chunks for '{source_path}'")


def _print_record(record) -> None:
    icon = {"ok": "[OK]", "skipped": "[SKIP]", "failed": "[FAIL]"}.get(record.status, "[?]")
    print(
        f"  {icon:<6}  [{record.status.upper():<7}]  "
        f"{record.document_title:<40}  "
        f"{record.chunk_count:>4} chunks  "
        f"{record.source_path}"
    )
    if record.error:
        print(f"          ERROR: {record.error}")


def main() -> None:
    args = _build_parser().parse_args()

    if args.status:
        _print_status()
        return

    if args.preview:
        target = args.file or "./data/policies"
        _run_preview(target)
        return

    if args.remove:
        from app.services.chroma_service import init_chroma
        init_chroma()
        _run_remove(args.remove)
        return

    # Default: initialize chroma then ingest
    from app.services.chroma_service import init_chroma
    init_chroma()

    if args.file:
        _run_ingest_file(args.file, args.manifest, args.force)
    else:
        _run_ingest_dir(
            args.dir, args.extensions, args.manifest, args.force
        )

    # Print final status
    _print_status()


if __name__ == "__main__":
    main()
