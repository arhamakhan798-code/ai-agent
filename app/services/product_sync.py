"""
app/services/product_sync.py
─────────────────────────────────────────────────────────────────────────────
Robust Product Catalog Synchronization Service.

Sources supported:
1. Google Sheets (live cloud sync via gspread)
2. Local CSV file (offline/development fallback)

Features:
- Column header normalization (handles common variations like "Price ($)", "Stock Qty").
- Strict type validation and cleaning (strips "$", "%", commas).
- Content-hash change detection: skips unnecessary DB writes if sheet has not changed.
- Incremental / batch upsert by SKU.
- Updates sync_states table with audit history.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.config import settings
from app.models.sync_state import SyncState
from app.repositories.product_repository import ProductRepository

logger = logging.getLogger(__name__)


class ProductSyncService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = ProductRepository(db)

    def sync(
        self,
        source: str = "google_sheets",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute catalog synchronization.

        source: "google_sheets" | "csv"
        force: If True, ignore content-hash check and force DB upsert.
        """
        logger.info("Starting product catalog sync from source: '%s' (force=%s)", source, force)

        try:
            if source == "google_sheets":
                raw_rows, content_hash = self._fetch_google_sheets()
            elif source == "csv":
                raw_rows, content_hash = self._fetch_csv()
            else:
                raise ValueError(f"Unknown sync source: '{source}'. Choose 'google_sheets' or 'csv'.")

            # Check if unchanged
            existing_state = (
                self.db.query(SyncState)
                .filter(SyncState.source_name == source)
                .first()
            )

            if not force and existing_state and existing_state.content_hash == content_hash and existing_state.status == "success":
                logger.info("Source '%s' has not changed (hash: %s). Skipping sync.", source, content_hash[:8])
                return {
                    "source": source,
                    "status": "skipped",
                    "rows_synced": existing_state.rows_synced,
                    "content_hash": content_hash,
                    "message": "Data unchanged since last sync.",
                }

            # Parse and clean items
            cleaned_items = self._clean_and_validate(raw_rows)
            upserted_count = self.repo.upsert_batch(cleaned_items)

            # Record success in sync_states
            now = datetime.utcnow()
            if existing_state:
                existing_state.last_sync_time = now
                existing_state.rows_synced = upserted_count
                existing_state.content_hash = content_hash
                existing_state.status = "success"
                existing_state.error_message = ""
            else:
                new_state = SyncState(
                    source_name=source,
                    last_sync_time=now,
                    rows_synced=upserted_count,
                    content_hash=content_hash,
                    status="success",
                    error_message="",
                )
                self.db.add(new_state)

            self.db.commit()

            logger.info("Successfully synced %d products from '%s'.", upserted_count, source)
            return {
                "source": source,
                "status": "success",
                "rows_synced": upserted_count,
                "content_hash": content_hash,
                "timestamp": now.isoformat(),
            }

        except Exception as exc:
            self.db.rollback()
            logger.exception("Catalog sync failed for source '%s': %s", source, exc)

            # Record failure in sync_states
            try:
                state = (
                    self.db.query(SyncState)
                    .filter(SyncState.source_name == source)
                    .first()
                )
                if not state:
                    state = SyncState(source_name=source)
                    self.db.add(state)
                state.status = "failed"
                state.error_message = str(exc)
                state.last_sync_time = datetime.utcnow()
                self.db.commit()
            except Exception:
                pass

            return {
                "source": source,
                "status": "failed",
                "error": str(exc),
            }

    # ── Fetching Data ─────────────────────────────────────────────────────────

    def _fetch_google_sheets(self) -> Tuple[List[Dict[str, Any]], str]:
        """Fetch rows from Google Sheets using gspread."""
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except ImportError:
            raise RuntimeError("gspread or google-auth not installed. Cannot sync from Google Sheets.")

        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]

        # Authenticate
        if settings.GOOGLE_SERVICE_ACCOUNT_JSON:
            info = json.loads(settings.GOOGLE_SERVICE_ACCOUNT_JSON)
            creds = Credentials.from_service_account_info(info, scopes=scope)
        elif os.path.exists(settings.GOOGLE_SERVICE_ACCOUNT_JSON_PATH):
            creds = Credentials.from_service_account_file(
                settings.GOOGLE_SERVICE_ACCOUNT_JSON_PATH, scopes=scope
            )
        else:
            raise FileNotFoundError(
                f"Google service account JSON not found at: {settings.GOOGLE_SERVICE_ACCOUNT_JSON_PATH}"
            )

        client = gspread.authorize(creds)
        sheet = client.open_by_key(settings.GOOGLE_SHEETS_SPREADSHEET_ID)
        worksheet = sheet.worksheet(settings.GOOGLE_SHEETS_SHEET_NAME)

        records = worksheet.get_all_records()
        content_str = json.dumps(records, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()

        logger.info("Fetched %d rows from Google Sheets worksheet '%s'", len(records), settings.GOOGLE_SHEETS_SHEET_NAME)
        return records, content_hash

    def _fetch_csv(self) -> Tuple[List[Dict[str, Any]], str]:
        """Fetch rows from local CSV backup."""
        csv_path = Path(settings.LOCAL_CSV_BACKUP_PATH).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Local CSV backup not found at: {csv_path}")

        records: List[Dict[str, Any]] = []
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))

        content_str = json.dumps(records, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode("utf-8")).hexdigest()

        logger.info("Fetched %d rows from CSV '%s'", len(records), csv_path.name)
        return records, content_hash

    # ── Cleaning and Normalization ────────────────────────────────────────────

    def _clean_and_validate(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize column keys and parse numeric types cleanly."""
        cleaned_list: List[Dict[str, Any]] = []

        for row in rows:
            # Normalize keys to lowercase stripped
            norm_row = {
                re.sub(r"[^a-z0-9]", "", str(k).lower()): v
                for k, v in row.items()
            }

            def _find_field(*candidates: str) -> Any:
                for cand in candidates:
                    for k, v in norm_row.items():
                        if cand in k:
                            return v
                return None

            sku = _find_field("sku", "itemid", "productid", "code")
            name = _find_field("name", "title", "product", "model")
            category = _find_field("category") or "General"
            brand = _find_field("brand") or "Generic"
            description = _find_field("description", "desc", "detail") or ""

            if not sku or not name:
                continue  # Skip invalid rows missing identity

            price = self._parse_float(_find_field("price", "cost") or 0.0)
            discount = self._parse_float(_find_field("discount") or 0.0)
            stock = self._parse_int(_find_field("stock", "quantity", "qty") or 0)

            cleaned_list.append({
                "sku": str(sku).strip(),
                "name": str(name).strip(),
                "category": str(category).strip(),
                "brand": str(brand).strip(),
                "price": price,
                "discount": discount,
                "stock": stock,
                "description": str(description).strip(),
                "is_active": True,
            })

        return cleaned_list

    @staticmethod
    def _parse_float(val: Any) -> float:
        if val is None or val == "":
            return 0.0
        cleaned = re.sub(r"[^\d.]", "", str(val))
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    @staticmethod
    def _parse_int(val: Any) -> int:
        if val is None or val == "":
            return 0
        cleaned = re.sub(r"[^\d]", "", str(val))
        try:
            return int(cleaned) if cleaned else 0
        except ValueError:
            return 0
