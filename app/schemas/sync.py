"""
app/schemas/sync.py
─────────────────────────────────────────────────────────────────────────────
Pydantic schemas for synchronization requests and responses.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class SyncTriggerRequest(BaseModel):
    source: str = Field("google_sheets", description="'google_sheets' | 'csv'")
    force: bool = Field(False, description="Sync even if content hash has not changed")


class SyncStatusResponse(BaseModel):
    source_name: str
    status: str
    rows_synced: int
    last_sync_time: Optional[str] = None
    content_hash: str = ""
    error_message: str = ""
