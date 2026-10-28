"""
app/api/v1/sync.py
─────────────────────────────────────────────────────────────────────────────
Catalog synchronization API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.sync_state import SyncState
from app.schemas.sync import SyncTriggerRequest, SyncStatusResponse
from app.services.product_sync import ProductSyncService

router = APIRouter()


@router.post("/sync", response_model=Dict[str, Any])
def trigger_sync(
    req: SyncTriggerRequest,
    db: Session = Depends(get_db),
):
    service = ProductSyncService(db)
    result = service.sync(source=req.source, force=req.force)
    return result


@router.get("/sync/status", response_model=List[SyncStatusResponse])
def get_sync_status(db: Session = Depends(get_db)):
    states = db.query(SyncState).all()
    return [
        SyncStatusResponse(
            source_name=s.source_name,
            status=s.status,
            rows_synced=s.rows_synced,
            last_sync_time=s.last_sync_time.isoformat() if s.last_sync_time else None,
            content_hash=s.content_hash,
            error_message=s.error_message,
        )
        for s in states
    ]
