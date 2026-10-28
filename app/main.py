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
from app.api.v1.chat import router as chat_router, handle_chat, ChatRequest
from app.api.v1.products import router as products_router
from app.api.v1.sync import router as sync_router
from app.api.v1.orders import router as orders_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    # Initial product catalog sync from CSV fallback if empty
    from app.core.database import get_db_context
    with get_db_context() as db:
        from app.repositories.product_repository import ProductRepository
        repo = ProductRepository(db)
        if repo.count() == 0 and os.path.exists(settings.LOCAL_CSV_BACKUP_PATH):
            logger.info("Product catalog empty. Seeding from local backup...")
            sync_svc = ProductSyncService(db)
            sync_svc.sync(source="csv")

    yield

    # Shutdown
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
    history_raw = body.get("conversation_history") or []
    chat_req = ChatRequest(
        message=msg,
        conversation_history=[],
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
def health_check():
    from app.services.chroma_service import get_status as chroma_status
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "chroma": chroma_status(),
    }


@app.get("/")
def serve_home():
    widget_path = Path("widget.html").resolve()
    if widget_path.exists():
        return FileResponse(widget_path)
    return {"message": f"{settings.PROJECT_NAME} API is running. See /docs for Swagger UI."}
