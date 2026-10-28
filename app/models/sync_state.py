"""
app/models/sync_state.py
─────────────────────────────────────────────────────────────────────────────
SQLAlchemy Model for Tracking Catalog Synchronization State.
"""

from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.core.database import Base


class SyncState(Base):
    __tablename__ = "sync_states"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    source_name = Column(String(50), nullable=False, unique=True, index=True)
    last_sync_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    rows_synced = Column(Integer, default=0, nullable=False)
    content_hash = Column(String(64), default="", nullable=False)
    status = Column(String(20), default="success", nullable=False)  # "success" | "failed"
    error_message = Column(Text, default="", nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_name": self.source_name,
            "last_sync_time": (
                self.last_sync_time.isoformat() if self.last_sync_time else None
            ),
            "rows_synced": self.rows_synced,
            "content_hash": self.content_hash,
            "status": self.status,
            "error_message": self.error_message,
        }
