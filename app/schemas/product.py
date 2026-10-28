"""
app/schemas/product.py
─────────────────────────────────────────────────────────────────────────────
Pydantic schemas for product validation, filtering, and API responses.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ProductBase(BaseModel):
    sku: str = Field(..., description="Unique Stock Keeping Unit")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    brand: str = Field(..., description="Brand name")
    price: float = Field(..., ge=0.0, description="Base retail price")
    discount: float = Field(0.0, ge=0.0, le=100.0, description="Discount percentage")
    stock: int = Field(0, ge=0, description="Current available inventory")
    description: str = Field("", description="Detailed product description")
    is_active: bool = Field(True, description="Whether product is actively sold")


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = Field(None, ge=0.0)
    discount: Optional[float] = Field(None, ge=0.0, le=100.0)
    stock: Optional[int] = Field(None, ge=0)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ProductResponse(ProductBase):
    id: int
    effective_price: float

    model_config = ConfigDict(from_attributes=True)


class ProductFilter(BaseModel):
    query: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    in_stock_only: bool = False
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)
