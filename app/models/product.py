"""
app/models/product.py
─────────────────────────────────────────────────────────────────────────────
SQLAlchemy Model for Products catalog.

Guarantees relational integrity for SKU, price, discount, and stock.
"""

from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    Boolean,
    DateTime,
    Index,
)
from app.core.database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    sku = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    brand = Column(String(100), nullable=False, index=True)
    price = Column(Float, nullable=False)
    discount = Column(Float, default=0.0, nullable=False)
    stock = Column(Integer, default=0, nullable=False)
    description = Column(Text, default="", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_products_category_brand", "category", "brand"),
        Index("ix_products_price_stock", "price", "stock"),
    )

    @property
    def effective_price(self) -> float:
        """Calculate final price after discount percentage."""
        if self.discount and self.discount > 0:
            return round(self.price * (1.0 - (self.discount / 100.0)), 2)
        return round(self.price, 2)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sku": self.sku,
            "name": self.name,
            "category": self.category,
            "brand": self.brand,
            "price": self.price,
            "discount": self.discount,
            "effective_price": self.effective_price,
            "stock": self.stock,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
