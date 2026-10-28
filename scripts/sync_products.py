"""
scripts/sync_products.py
─────────────────────────────────────────────────────────────────────────────
CLI for synchronizing product catalog from Google Sheets or local CSV into the
relational database.

Usage examples:
    # Sync from Google Sheets (default):
    python -m scripts.sync_products

    # Sync from local CSV:
    python -m scripts.sync_products --source csv

    # Force re-sync even if content hash hasn't changed:
    python -m scripts.sync_products --force

    # View catalog sync status and product counts:
    python -m scripts.sync_products --status
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sync_products")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize Product Catalog into Relational Database")
    parser.add_argument(
        "--source",
        choices=["google_sheets", "csv"],
        default="google_sheets",
        help="Data source to pull from (default: google_sheets)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force sync even if source data hash is unchanged",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current catalog count and sync status",
    )
    args = parser.parse_args()

    from app.core.database import init_db, get_db_context
    from app.models.sync_state import SyncState
    from app.repositories.product_repository import ProductRepository
    from app.services.product_sync import ProductSyncService

    init_db()

    with get_db_context() as db:
        if args.status:
            repo = ProductRepository(db)
            count = repo.count()
            states = db.query(SyncState).all()
            print("\n-- Product Catalog Status -------------------------")
            print(f"Total Products in DB: {count}")
            print("\nSync History:")
            for s in states:
                print(
                    f"  Source: {s.source_name:<15} | Status: {s.status:<8} | "
                    f"Rows: {s.rows_synced:<5} | Last Sync: {s.last_sync_time}"
                )
                if s.error_message:
                    print(f"    Error: {s.error_message}")
            print("---------------------------------------------------\n")
            return

        sync_service = ProductSyncService(db)
        result = sync_service.sync(source=args.source, force=args.force)

        print("\n-- Sync Result ------------------------------------")
        for k, v in result.items():
            print(f"  {k:<15}: {v}")
        print("---------------------------------------------------\n")


if __name__ == "__main__":
    main()
