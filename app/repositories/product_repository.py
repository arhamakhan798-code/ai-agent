"""
app/repositories/product_repository.py
─────────────────────────────────────────────────────────────────────────────
Repository pattern for high-performance relational product catalog operations.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, List
from sqlalchemy import or_, and_, func, case
from sqlalchemy.orm import Session

from app.models.product import Product
from app.schemas.product import ProductCreate, ProductFilter

logger = logging.getLogger(__name__)


class ProductRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, product_id: int) -> Optional[Product]:
        return self.db.query(Product).filter(Product.id == product_id).first()

    def get_by_sku(self, sku: str) -> Optional[Product]:
        if not sku:
            return None
        return (
            self.db.query(Product)
            .filter(func.lower(Product.sku) == sku.strip().lower())
            .first()
        )

    def get_by_exact_or_partial_name(self, name: str) -> Optional[Product]:
        """Find product by exact name or closest substring."""
        if not name:
            return None
        name_clean = name.strip().lower()
        # Try exact first
        prod = (
            self.db.query(Product)
            .filter(func.lower(Product.name) == name_clean)
            .first()
        )
        if prod:
            return prod
        # Fallback to ILIKE substring
        return (
            self.db.query(Product)
            .filter(Product.name.ilike(f"%{name_clean}%"))
            .first()
        )

    def search(
        self,
        filters: Optional[ProductFilter] = None,
        query: Optional[str] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        in_stock_only: bool = False,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Product]:
        """Flexible relational search with SQL indexing."""
        if filters:
            query = filters.query or query
            category = filters.category or category
            brand = filters.brand or brand
            min_price = filters.min_price or min_price
            max_price = filters.max_price or max_price
            in_stock_only = filters.in_stock_only or in_stock_only
            limit = filters.limit or limit
            offset = filters.offset or offset

        stmt = self.db.query(Product).filter(Product.is_active == True)

        if query:
            q_clean = query.strip().lower()
            stopwords = {
                "show", "me", "a", "an", "the", "and", "or", "for", "in", "of", "to",
                "with", "any", "some", "what", "do", "you", "have", "can", "i", "get",
                "please", "recommend", "suggest", "options", "alternatives", "best",
            }
            raw_terms = re.findall(r"\b[a-z0-9]+\b", q_clean)
            meaningful_terms = [t for t in raw_terms if t not in stopwords and len(t) > 1]
            search_terms = meaningful_terms if meaningful_terms else raw_terms

            if search_terms:
                # Add singular/plural variants for common queries (e.g., phones -> phone, laptops -> laptop, mice -> mouse)
                expanded_terms = set(search_terms)
                irregular_plurals = {
                    "mice": "mouse",
                    "mouse": "mice",
                }
                for t in list(search_terms):
                    if t in irregular_plurals:
                        expanded_terms.add(irregular_plurals[t])
                    elif t.endswith("ies") and len(t) > 4:
                        expanded_terms.add(t[:-3] + "y")
                    elif t.endswith("es") and len(t) > 3:
                        expanded_terms.add(t[:-2])
                    elif t.endswith("s") and len(t) > 2:
                        expanded_terms.add(t[:-1])

                search_terms = list(expanded_terms)

                or_clauses = []
                for term in expanded_terms:
                    pattern = f"%{term}%"
                    or_clauses.extend([
                        Product.name.ilike(pattern),
                        Product.category.ilike(pattern),
                        Product.brand.ilike(pattern),
                        Product.description.ilike(pattern),
                        Product.sku.ilike(pattern),
                    ])
                stmt = stmt.filter(or_(*or_clauses))

        if category:
            stmt = stmt.filter(Product.category.ilike(f"%{category.strip()}%"))

        if brand:
            stmt = stmt.filter(Product.brand.ilike(f"%{brand.strip()}%"))

        if min_price is not None:
            stmt = stmt.filter(Product.price >= min_price)

        if max_price is not None:
            stmt = stmt.filter(Product.price <= max_price)

        if in_stock_only:
            stmt = stmt.filter(Product.stock > 0)

        # Prioritize matches where category or name contains the term, over description matches
        order_clauses = []
        if query and search_terms:
            conditions = []
            for t in search_terms:
                conditions.append((func.lower(Product.category) == t, 1))
                conditions.append((Product.category.ilike(f"{t}%"), 2))
                conditions.append((Product.category.ilike(f"%{t}%"), 3))
                conditions.append((func.lower(Product.name) == t, 4))
                conditions.append((Product.name.ilike(f"{t}%"), 5))
                conditions.append((Product.name.ilike(f"%{t}%"), 6))
            order_clauses.append(case(*conditions, else_=7))
        order_clauses.append(Product.name.asc())

        return (
            stmt.order_by(*order_clauses)
            .offset(offset)
            .limit(limit)
            .all()
        )

    def upsert_batch(self, items: List[dict]) -> int:
        """
        Upsert a batch of product dictionaries by SKU.
        If SKU exists, update all fields.
        If SKU does not exist, insert new record.
        """
        if not items:
            return 0

        upserted_count = 0
        for item in items:
            sku = (item.get("sku") or "").strip()
            if not sku:
                continue

            existing = (
                self.db.query(Product)
                .filter(func.lower(Product.sku) == sku.lower())
                .first()
            )

            if existing:
                existing.name = item.get("name", existing.name)
                existing.category = item.get("category", existing.category)
                existing.brand = item.get("brand", existing.brand)
                existing.price = float(item.get("price", existing.price))
                existing.discount = float(item.get("discount", existing.discount))
                existing.stock = int(item.get("stock", existing.stock))
                existing.description = item.get("description", existing.description)
                existing.is_active = item.get("is_active", existing.is_active)
            else:
                new_prod = Product(
                    sku=sku,
                    name=item.get("name", "Unnamed Product"),
                    category=item.get("category", "General"),
                    brand=item.get("brand", "Unknown"),
                    price=float(item.get("price", 0.0)),
                    discount=float(item.get("discount", 0.0)),
                    stock=int(item.get("stock", 0)),
                    description=item.get("description", ""),
                    is_active=item.get("is_active", True),
                )
                self.db.add(new_prod)

            upserted_count += 1

        self.db.commit()
        return upserted_count

    def count(self) -> int:
        return self.db.query(Product).count()

    def get_categories(self) -> List[str]:
        results = (
            self.db.query(Product.category)
            .distinct()
            .order_by(Product.category.asc())
            .all()
        )
        return [r[0] for r in results if r[0]]

    def get_brands(self) -> List[str]:
        results = (
            self.db.query(Product.brand)
            .distinct()
            .order_by(Product.brand.asc())
            .all()
        )
        return [r[0] for r in results if r[0]]

    def get_all_names_sample(self, limit: int = 50) -> List[str]:
        results = self.db.query(Product.name).limit(limit).all()
        return [r[0] for r in results if r[0]]
