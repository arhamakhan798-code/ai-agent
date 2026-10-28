"""
app/core/database.py
─────────────────────────────────────────────────────────────────────────────
Database engine, session management, and base declarative class.

Supports:
- PostgreSQL (Production / Staging) via psycopg2
- SQLite fallback (Local Development / Testing) when PostgreSQL is not configured
  or unreachable.

Ensures strict connection pooling and thread-safe session handling.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


def _get_engine():
    db_url = settings.DATABASE_URL
    connect_args = {}

    # If PostgreSQL URL is localhost default and SQLITE_FALLBACK is True,
    # or if the url starts with sqlite, handle sqlite options
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":///" in db_url:
            path_part = db_url.split(":///", 1)[1]
            if path_part and not path_part.startswith(":memory:"):
                import os
                os.makedirs(os.path.dirname(os.path.abspath(path_part)), exist_ok=True)
        return create_engine(db_url, connect_args=connect_args)

    try:
        connect_args["connect_timeout"] = 3
        engine = create_engine(
            db_url,
            connect_args=connect_args,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
        )
        # Test connection with short timeout
        with engine.connect() as conn:
            pass
        logger.info("Connected to PostgreSQL database successfully.")
        return engine
    except Exception as exc:
        if settings.SQLITE_FALLBACK:
            sqlite_url = f"sqlite:///{settings.SQLITE_DB_PATH}"
            logger.warning(
                "Could not connect to PostgreSQL (%s). Falling back to SQLite at: %s",
                exc,
                sqlite_url,
            )
            connect_args["check_same_thread"] = False
            return create_engine(sqlite_url, connect_args=connect_args)
        raise exc


engine = _get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all registered tables in the database."""
    # Ensure all models are imported before creating tables
    import app.models.product
    import app.models.sync_state
    import app.models.order

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for scripts and background services."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
