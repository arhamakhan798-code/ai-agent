"""
app/api/v1/chat.py
─────────────────────────────────────────────────────────────────────────────
Chat and RAG query API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.rag_engine import RAGEngine

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question or request")
    conversation_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous chat turns")
    session_id: Optional[str] = Field(default="", description="Client session identifier")


class ChatResponse(BaseModel):
    reply: str
    intent: str
    products: List[Dict[str, Any]] = []
    matching_products: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    product_count: int = 0
    status: str = "success"


@router.post("/chat", response_model=ChatResponse)
def handle_chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
):
    history_dicts = [{"role": m.role, "content": m.content} for m in req.conversation_history or []]
    engine = RAGEngine(db)
    result = engine.process_query(
        user_message=req.message,
        conversation_history=history_dicts,
    )
    return ChatResponse(**result)
