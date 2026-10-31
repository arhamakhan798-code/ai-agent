"""
app/api/v1/orders.py
─────────────────────────────────────────────────────────────────────────────
Order checkout, status lookup, and refund request API endpoints.
Ensures full compatibility with widget.html checkout modals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from fastapi import APIRouter, Request, HTTPException

from app.services.order_service import (
    place_order_service,
    view_order_service,
    request_refund_service,
    sanitize_input,
    _IBAN_RE,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/order/place")
async def place_order_endpoint(request: Request):
    """Delegate to order placement service with order date tracking."""
    body = await request.json()
    c_name = sanitize_input(body.get("customer_name") or "", max_len=100)
    c_phone = sanitize_input(body.get("contact_number") or "", max_len=30)
    c_email = sanitize_input(body.get("email") or "", max_len=100)
    c_bank = sanitize_input(body.get("bank") or "", max_len=100)
    c_iban = sanitize_input(body.get("iban") or "", max_len=50)
    c_address = sanitize_input(body.get("customer_address") or "", max_len=200)

    items = body.get("items", [])
    single_prod = sanitize_input(body.get("product") or "", max_len=300)

    res = place_order_service(
        customer_name=c_name,
        contact_number=c_phone,
        email=c_email,
        bank=c_bank,
        iban=c_iban,
        customer_address=c_address,
        product_name_or_sku=single_prod,
        items=items,
    )
    if "error" in res:
        raise HTTPException(status_code=400, detail=res.get("error"))
    return res


@router.post("/order/lookup")
@router.post("/order/view")
async def view_order_endpoint(request: Request):
    """View order status by order number and secret passkey."""
    body = await request.json()
    order_num = sanitize_input(body.get("order_number") or "", max_len=20)
    passkey = sanitize_input(body.get("passkey") or "", max_len=30)

    if not order_num or not passkey:
        raise HTTPException(status_code=400, detail="Both 'order_number' and 'passkey' are required.")

    res = view_order_service(order_num, passkey)
    if "error" in res:
        raise HTTPException(status_code=403, detail=res.get("error"))
    return res


@router.post("/order/refund")
async def refund_order_endpoint(request: Request):
    """Request refund with passkey verification and date recording."""
    body = await request.json()
    order_num = sanitize_input(body.get("order_number") or "", max_len=20)
    passkey = sanitize_input(body.get("passkey") or "", max_len=30)
    days = int(body.get("days_since_purchase", 5))
    orig = bool(body.get("is_original_condition", True))
    reason = sanitize_input(body.get("reason", "Customer return request"), max_len=500)

    if not order_num or not passkey:
        raise HTTPException(status_code=400, detail="Both 'order_number' and 'passkey' are required.")

    res = request_refund_service(order_num, passkey, days, orig, reason)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res.get("error"))
    return res


@router.post("/session/reset")
async def session_reset_endpoint(request: Request):
    """Reset session state."""
    return {"status": "ok", "message": "Session reset successfully."}
