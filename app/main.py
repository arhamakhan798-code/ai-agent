"""
app/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI Production Application Entry Point.

Features:
- Lifespan initialization: initializes SQLite/PostgreSQL tables and ChromaDB.
- Full CORS middleware support.
- Routes: /api/v1/chat, /api/v1/products, /api/v1/sync, /health
- Backward-compatible /chat route for widget.html integration.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.core.database import init_db, get_db
from app.services.chroma_service import init_chroma
from app.services.policy_ingestion import ingest_directory
from app.services.product_sync import ProductSyncService
from app.api.v1.chat import router as chat_router, handle_chat, ChatRequest, ChatMessage
from app.api.v1.products import router as products_router
from app.api.v1.sync import router as sync_router
from app.api.v1.orders import router as orders_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger("app.main")


async def _sheets_sync_loop():
    """
    Background task: polls Google Sheets every SYNC_POLL_INTERVAL_SECONDS seconds.
    Uses content-hash comparison inside ProductSyncService — only writes to DB
    when the sheet actually changed, so this is cheap on idle.
    """
    interval = max(settings.SYNC_POLL_INTERVAL_SECONDS, 10)  # minimum 10s
    logger.info(
        "Google Sheets real-time sync started — polling every %ds for changes.", interval
    )
    while True:
        await asyncio.sleep(interval)
        try:
            from app.core.database import get_db_context
            with get_db_context() as db:
                svc = ProductSyncService(db)
                result = svc.sync(source="google_sheets")
                if result.get("status") == "success":
                    logger.info(
                        "Sheets sync: %d products upserted (hash %s…)",
                        result.get("rows_synced", 0),
                        str(result.get("content_hash", ""))[:8],
                    )
                elif result.get("status") == "skipped":
                    logger.debug("Sheets sync: no changes detected.")
                else:
                    logger.warning("Sheets sync returned: %s", result)
        except Exception as exc:
            logger.warning("Sheets sync poll failed (will retry): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Railway volume path adjustment ────────────────────────────────────────
    # When RAILWAY_VOLUME_MOUNT_PATH is set (Railway attaches a persistent volume),
    # relocate SQLite and ChromaDB into that directory so data survives redeploys.
    volume_root = settings.RAILWAY_VOLUME_MOUNT_PATH.strip()
    if volume_root:
        volume_path = Path(volume_root)
        (volume_path / "data").mkdir(parents=True, exist_ok=True)
        (volume_path / "chroma_db").mkdir(parents=True, exist_ok=True)
        (volume_path / "data" / "policies").mkdir(parents=True, exist_ok=True)
        # Override runtime paths on the singleton settings object
        settings.SQLITE_DB_PATH = str(volume_path / "data" / "techmart_local.db")
        settings.DATABASE_URL = f"sqlite:///{settings.SQLITE_DB_PATH}"
        settings.CHROMA_PERSIST_DIRECTORY = str(volume_path / "chroma_db")
        logger.info("Railway volume detected. Paths relocated under: %s", volume_root)
    else:
        # Ensure local data dirs exist (first run / Docker)
        Path("./data").mkdir(parents=True, exist_ok=True)
        Path("./chroma_db").mkdir(parents=True, exist_ok=True)

    # Startup: Initialize Database & Vector Store
    logger.info("Initializing relational database tables...")
    init_db()

    logger.info("Initializing ChromaDB persistent collection...")
    init_chroma()

    # Pre-index policies if present
    policies_dir = Path("./data/policies")
    if policies_dir.exists():
        try:
            logger.info("Checking policy directory for updates...")
            ingest_directory(policies_dir)
        except Exception as exc:
            logger.warning("Initial policy ingestion notice: %s", exc)

    # Initial product catalog sync — try Google Sheets first, fall back to CSV
    from app.core.database import get_db_context
    with get_db_context() as db:
        svc = ProductSyncService(db)
        try:
            logger.info("Initial sync: fetching product catalog from Google Sheets...")
            result = svc.sync(source="google_sheets")
            if result.get("status") in ("success", "skipped"):
                logger.info(
                    "Google Sheets initial sync OK — %d products in catalog.",
                    result.get("rows_synced", 0),
                )
            else:
                raise RuntimeError(result.get("error", "unknown"))
        except Exception as exc:
            logger.warning(
                "Google Sheets sync failed on startup (%s). Falling back to CSV.", exc
            )
            from app.repositories.product_repository import ProductRepository
            repo = ProductRepository(db)
            if repo.count() == 0 and os.path.exists(settings.LOCAL_CSV_BACKUP_PATH):
                logger.info("Product catalog empty. Seeding from local CSV backup...")
                svc.sync(source="csv")

    # Launch real-time Google Sheets background polling loop
    sync_task = asyncio.create_task(_sheets_sync_loop())
    logger.info("Real-time Google Sheets sync background task started.")

    yield

    # Shutdown — cancel the polling loop cleanly
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-Ready RAG Chatbot for Products and Company Policies",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers (v1 and root /api aliases for widget.html)
app.include_router(chat_router, prefix=settings.API_V1_STR, tags=["Chat"])
app.include_router(products_router, prefix=settings.API_V1_STR, tags=["Products"])
app.include_router(sync_router, prefix=settings.API_V1_STR, tags=["Sync"])
app.include_router(orders_router, prefix=settings.API_V1_STR, tags=["Orders"])

# Direct /api/* aliases for widget.html
app.include_router(chat_router, prefix="/api", tags=["Legacy Chat"])
app.include_router(orders_router, prefix="/api", tags=["Legacy Orders"])


# ── Backward-compatible routes for widget.html ───────────────────────────────

@app.post("/chat")
async def legacy_chat(req: Request, db: Session = Depends(get_db)):
    """Handles requests from widget.html sending {message: '...' or message: ''}."""
    body = await req.json()
    msg = body.get("message") or body.get("user_message") or ""
    # Accept both field names — widget.html sends "history", v1 API sends "conversation_history"
    history_raw = body.get("conversation_history") or body.get("history") or []

    def _normalize_role(role: str) -> str:
        """widget.html uses 'agent' for bot — normalize to 'assistant'."""
        if role in ("agent", "bot", "clara", "assistant", "model"):
            return "assistant"
        return "user"

    parsed_history = [
        ChatMessage(role=_normalize_role(h.get("role", "user")), content=h.get("content", ""))
        for h in history_raw
        if h.get("content")
    ]
    chat_req = ChatRequest(
        message=msg,
        conversation_history=parsed_history,
        session_id=body.get("session_id", ""),
    )
    res = handle_chat(chat_req, db=db)

    # Return structure compatible with widget.html
    return {
        "reply": res.reply,
        "intent": res.intent,
        "products": res.products,
        "matching_products": res.products,
        "sources": res.sources,
        "status": "success",
    }


@app.get("/health")
@app.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    from app.services.chroma_service import get_status as chroma_status
    from app.models.product import Product
    from app.models.order import Order
    try:
        prod_count = db.query(Product).count()
    except Exception:
        prod_count = 0
    try:
        order_count = db.query(Order).count()
    except Exception:
        order_count = 0
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "chroma": chroma_status(),
        "products": prod_count,
        "orders": order_count,
    }


@app.get("/")
def serve_home():
    widget_path = Path("widget.html").resolve()
    if widget_path.exists():
        return FileResponse(widget_path)
    return {"message": f"{settings.PROJECT_NAME} API is running. See /docs for Swagger UI."}
