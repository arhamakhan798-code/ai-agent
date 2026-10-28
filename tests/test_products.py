"""
tests/test_products.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive unit and integration tests for Product Catalog & Sync Pipeline.

Tests:
- Product model calculation & validation
- ProductRepository CRUD, search, filter, and batch upsert
- ProductSyncService with CSV fallback & content hashing
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.product import Product
from app.models.sync_state import SyncState
from app.repositories.product_repository import ProductRepository
from app.services.product_sync import ProductSyncService
from app.schemas.product import ProductFilter


@pytest.fixture
def db_session():
    """Isolated in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


class TestProductModel:
    def test_effective_price_with_discount(self):
        p = Product(sku="SKU-1", name="Test Item", category="Tech", brand="Acme", price=100.0, discount=15.0, stock=5)
        assert p.effective_price == 85.0

    def test_effective_price_without_discount(self):
        p = Product(sku="SKU-2", name="Test Item 2", category="Tech", brand="Acme", price=250.0, discount=0.0, stock=5)
        assert p.effective_price == 250.0

    def test_to_dict(self):
        p = Product(sku="SKU-3", name="Phone", category="Mobile", brand="BrandX", price=500.0, discount=10.0, stock=20)
        d = p.to_dict()
        assert d["sku"] == "SKU-3"
        assert d["effective_price"] == 450.0
        assert d["stock"] == 20


class TestProductRepository:
    def test_upsert_batch_and_get_by_sku(self, db_session):
        repo = ProductRepository(db_session)
        items = [
            {"sku": "SKU-A", "name": "Alpha Item", "category": "Gadgets", "brand": "BrandA", "price": 50.0, "discount": 0.0, "stock": 10},
            {"sku": "SKU-B", "name": "Beta Item", "category": "Gadgets", "brand": "BrandB", "price": 100.0, "discount": 20.0, "stock": 0},
        ]
        count = repo.upsert_batch(items)
        assert count == 2
        assert repo.count() == 2

        prod_a = repo.get_by_sku("sku-a")  # Case insensitive
        assert prod_a is not None
        assert prod_a.name == "Alpha Item"

        # Update SKU-A
        items_update = [
            {"sku": "SKU-A", "name": "Alpha Item Updated", "category": "Gadgets", "brand": "BrandA", "price": 55.0, "discount": 10.0, "stock": 12},
        ]
        repo.upsert_batch(items_update)
        assert repo.count() == 2  # No duplicate
        updated_a = repo.get_by_sku("SKU-A")
        assert updated_a.name == "Alpha Item Updated"
        assert updated_a.price == 55.0
        assert updated_a.effective_price == 49.5

    def test_search_and_filter(self, db_session):
        repo = ProductRepository(db_session)
        items = [
            {"sku": "LAP-01", "name": "Dell XPS 15", "category": "Laptops", "brand": "Dell", "price": 1200.0, "discount": 10.0, "stock": 5},
            {"sku": "LAP-02", "name": "Apple MacBook Air", "category": "Laptops", "brand": "Apple", "price": 999.0, "discount": 0.0, "stock": 0},
            {"sku": "PHO-01", "name": "Apple iPhone 15", "category": "Phones", "brand": "Apple", "price": 899.0, "discount": 5.0, "stock": 15},
        ]
        repo.upsert_batch(items)

        # Keyword search
        res = repo.search(query="Apple")
        assert len(res) == 2

        # In stock only
        res_stock = repo.search(query="Apple", in_stock_only=True)
        assert len(res_stock) == 1
        assert res_stock[0].sku == "PHO-01"

        # Price range
        res_price = repo.search(min_price=1000.0)
        assert len(res_price) == 1
        assert res_price[0].sku == "LAP-01"

        # Category filter
        res_cat = repo.search(category="Laptops")
        assert len(res_cat) == 2

    def test_get_by_exact_or_partial_name(self, db_session):
        repo = ProductRepository(db_session)
        repo.upsert_batch([
            {"sku": "PS5-01", "name": "Sony PlayStation 5 Slim", "category": "Gaming", "brand": "Sony", "price": 449.99, "discount": 15.0, "stock": 10}
        ])

        # Exact match
        p1 = repo.get_by_exact_or_partial_name("Sony PlayStation 5 Slim")
        assert p1 is not None

        # Partial match
        p2 = repo.get_by_exact_or_partial_name("PlayStation 5")
        assert p2 is not None
        assert p2.sku == "PS5-01"


class TestProductSyncService:
    def test_sync_from_csv(self, db_session, monkeypatch, tmp_path):
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "Product Name,SKU,Category,Brand,Price ($),Discount (%),Stock Quantity,Description\n"
            "Test Laptop,TEST-SKU-1,Laptops,BrandX,$999.99,10%,15,A powerful laptop\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("app.config.settings.LOCAL_CSV_BACKUP_PATH", str(csv_file))

        service = ProductSyncService(db_session)
        res = service.sync(source="csv")

        assert res["status"] == "success"
        assert res["rows_synced"] == 1

        repo = ProductRepository(db_session)
        p = repo.get_by_sku("TEST-SKU-1")
        assert p is not None
        assert p.name == "Test Laptop"
        assert p.price == 999.99
        assert p.discount == 10.0
        assert p.stock == 15

        # Second sync with same file should skip
        res2 = service.sync(source="csv")
        assert res2["status"] == "skipped"

        # Force sync should run
        res3 = service.sync(source="csv", force=True)
        assert res3["status"] == "success"
