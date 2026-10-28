"""
app/models/order.py
─────────────────────────────────────────────────────────────────────────────
SQLAlchemy Models for Orders and Refunds tracking.
Keeps structured records synchronized with relational DB and Google Sheets.
"""

from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from app.core.database import Base


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    customer_name = Column(String(255), nullable=False)
    customer_email = Column(String(255), nullable=False, index=True)
    product_sku = Column(String(64), nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    quantity = Column(Integer, default=1, nullable=False)
    total_amount = Column(Float, nullable=False)
    order_date = Column(String(50), nullable=False)  # ISO or formatted date string
    status = Column(String(50), default="Confirmed", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "customer_name": self.customer_name,
            "customer_email": self.customer_email,
            "product_sku": self.product_sku,
            "product_name": self.product_name,
            "quantity": self.quantity,
            "total_amount": self.total_amount,
            "order_date": self.order_date,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Refund(Base):
    __tablename__ = "refunds"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(String(50), nullable=False, index=True)
    customer_email = Column(String(255), nullable=False, index=True)
    reason = Column(Text, default="", nullable=False)
    refund_request_date = Column(String(50), nullable=False)
    status = Column(String(50), default="Pending", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "customer_email": self.customer_email,
            "reason": self.reason,
            "refund_request_date": self.refund_request_date,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
