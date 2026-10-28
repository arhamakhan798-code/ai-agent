"""
tests/test_api.py
─────────────────────────────────────────────────────────────────────────────
End-to-End API integration tests using FastAPI TestClient.

Tests:
1. Health check & status endpoints
2. Product search & filter REST APIs
3. Sync audit API
4. RAG Chat:
   - Discount question -> direct answer, NO product cards (fixes the sticky card bug!)
   - Recommendation question -> answers + product cards returned
   - Policy question -> ChromaDB grounded answer + sources citation
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.core.database import get_db_context
from app.repositories.product_repository import ProductRepository


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module", autouse=True)
def seed_test_catalog():
    """Ensure catalog has standard items for API tests."""
    from app.core.database import init_db
    init_db()
    with get_db_context() as db:
        repo = ProductRepository(db)
        repo.upsert_batch([
            {
                "sku": "PS5-SLIM",
                "name": "Sony PlayStation 5 Slim",
                "category": "Gaming",
                "brand": "Sony",
                "price": 449.99,
                "discount": 15.0,
                "stock": 20,
                "description": "Next-gen console with ultra-fast SSD",
            },
            {
                "sku": "MAC-M3",
                "name": "Apple MacBook Air M3",
                "category": "Laptops",
                "brand": "Apple",
                "price": 1099.99,
                "discount": 0.0,
                "stock": 10,
                "description": "Lightweight laptop with M3 chip",
            },
        ])


class TestEndpoints:
    def test_health_check(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"

    def test_list_products(self, client):
        res = client.get("/api/v1/products")
        assert res.status_code == 200
        data = res.json()
        assert len(data) >= 2
        skus = [p["sku"] for p in data]
        assert "PS5-SLIM" in skus

    def test_filter_products_by_category(self, client):
        res = client.get("/api/v1/products?category=Gaming")
        assert res.status_code == 200
        data = res.json()
        assert all(p["category"] == "Gaming" for p in data)

    def test_get_product_by_sku(self, client):
        res = client.get("/api/v1/products/PS5-SLIM")
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "Sony PlayStation 5 Slim"
        assert data["discount"] == 15.0
        assert data["effective_price"] == 382.49

    def test_get_product_by_invalid_sku_404(self, client):
        res = client.get("/api/v1/products/UNKNOWN-SKU-999")
        assert res.status_code == 404

    def test_sync_status(self, client):
        res = client.get("/api/v1/sync/status")
        assert res.status_code == 200
        assert isinstance(res.json(), list)


class TestRAGChatRouting:
    def test_discount_question_returns_no_product_cards(self, client):
        """
        REGRESSION TEST FOR THE CORE BUG:
        Asking "can I get any more discounts in sony playstation slim"
        must answer directly and MUST NOT return product cards!
        """
        with patch("app.services.llm_service.llm_service.generate", return_value="The Sony PlayStation 5 Slim already has 15% off. There are no additional discounts at this time."):
            res = client.post(
                "/api/v1/chat",
                json={"message": "can I get any more discounts in sony playstation slim"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["intent"] == "product_question"
            assert "15%" in data["reply"]
            # CRITICAL: Product cards must NOT be returned for discount/price questions!
            assert len(data["products"]) == 0

    def test_recommendation_question_returns_product_cards(self, client):
        """
        Asking for recommendations ("show me laptops")
        MUST return product cards!
        """
        with patch("app.services.llm_service.llm_service.generate", return_value="Here are top recommended laptops:"):
            res = client.post(
                "/api/v1/chat",
                json={"message": "show me laptops and gaming consoles"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["intent"] == "recommend"
            assert len(data["products"]) > 0

    def test_policy_question_cites_provenance(self, client):
        """
        Asking about return window retrieves policy sources.
        """
        with patch("app.services.llm_service.llm_service.generate", return_value="Customers may return electronics within 15 days according to our return policy."):
            res = client.post(
                "/api/v1/chat",
                json={"message": "What is the return window for electronics?"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["intent"] in ("policy", "mixed")
            assert len(data["sources"]) > 0

    def test_legacy_widget_route(self, client):
        """
        Ensure /api/chat works cleanly with widget.html payload format.
        """
        with patch("app.services.llm_service.llm_service.generate", return_value="We are here to help!"):
            res = client.post(
                "/api/chat",
                json={"message": "Hello Clara", "session_id": "sess_test123"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "success"
            assert "reply" in data
