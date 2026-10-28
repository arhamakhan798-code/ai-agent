"""
app/api/v1/products.py
─────────────────────────────────────────────────────────────────────────────
Product catalog REST API endpoints.
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.product_repository import ProductRepository
from app.schemas.product import ProductResponse, ProductFilter

router = APIRouter()


@router.get("/products", response_model=List[ProductResponse])
def list_products(
    query: Optional[str] = Query(None, description="Search term for name, category, or brand"),
    category: Optional[str] = Query(None, description="Filter by category"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    min_price: Optional[float] = Query(None, ge=0.0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0.0, description="Maximum price"),
    in_stock_only: bool = Query(False, description="Filter for products with stock > 0"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    repo = ProductRepository(db)
    filters = ProductFilter(
        query=query,
        category=category,
        brand=brand,
        min_price=min_price,
        max_price=max_price,
        in_stock_only=in_stock_only,
        limit=limit,
        offset=offset,
    )
    products = repo.search(filters=filters)
    return products


@router.get("/products/categories", response_model=List[str])
def get_categories(db: Session = Depends(get_db)):
    repo = ProductRepository(db)
    return repo.get_categories()


@router.get("/products/brands", response_model=List[str])
def get_brands(db: Session = Depends(get_db)):
    repo = ProductRepository(db)
    return repo.get_brands()


@router.get("/products/{sku}", response_model=ProductResponse)
def get_product_by_sku(sku: str, db: Session = Depends(get_db)):
    repo = ProductRepository(db)
    product = repo.get_by_sku(sku)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product with SKU '{sku}' not found.")
    return product
