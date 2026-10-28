"""
app/services/rag_engine.py
─────────────────────────────────────────────────────────────────────────────
Hybrid RAG Engine: Products (Relational DB) + Policies (ChromaDB Vector Store).

Key Architectural Pillars:
1. Two Separate Knowledge Pipelines:
   - Strict factual product catalog from PostgreSQL/SQLite.
   - Semantic policy retrieval from ChromaDB with provenance metadata.
2. Structure-Aware Routing:
   - Accurately classifies query intent (product_question, policy, recommend,
     order_lookup, refund, mixed).
   - Eliminates "sticky recommendation mode": a follow-up question about discounts
     answers the discount directly instead of repeatedly showing product cards.
3. Hallucination Defense:
   - Product pricing, discounts, and inventory are injected as verified facts.
   - Policies are cited directly with section references.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.config import settings
from app.repositories.product_repository import ProductRepository
from app.services.chroma_service import query_policy
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, db: Session):
        self.db = db
        self.repo = ProductRepository(db)

    def process_query(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Main query processing pipeline:
        1. Classify intent
        2. Retrieve grounded context (relational DB and/or ChromaDB vector store)
        3. Generate response with LLM
        4. Package payload with product cards (only if intent is recommendation/search)
        """
        history = conversation_history or []
        msg_clean = user_message.strip()

        # 1. Routing & Intent Classification
        intent_info = self._classify_intent(msg_clean, history)
        intent = intent_info["intent"]
        product_ref = intent_info.get("product", "")

        logger.info("Classified intent='%s', product_ref='%s' for query: '%s'", intent, product_ref, msg_clean)

        # 2. Context Retrieval
        policy_chunks: List[Dict[str, Any]] = []
        matching_products: List[Dict[str, Any]] = []

        # Policy Retrieval (triggered for policy, mixed, or general questions)
        if intent in ("policy", "mixed", "general"):
            policy_chunks = query_policy(
                msg_clean,
                n_results=settings.POLICY_TOP_K,
                score_threshold=settings.POLICY_SCORE_THRESHOLD,
            )

        # Product Retrieval (triggered for product_question, recommend, mixed)
        if intent in ("product_question", "recommend", "mixed"):
            query_term = product_ref if product_ref else msg_clean
            products = self.repo.search(query=query_term, limit=settings.PRODUCT_TOP_K)
            if not products and product_ref:
                # Fallback to broader search
                products = self.repo.search(query=msg_clean, limit=settings.PRODUCT_TOP_K)
            matching_products = [p.to_dict() for p in products]

        # 3. Prompt Assembly & Grounded Generation
        system_prompt = self._build_system_prompt(
            intent=intent,
            policy_chunks=policy_chunks,
            products=matching_products,
        )

        llm_reply = llm_service.generate(
            system_prompt=system_prompt,
            user_message=msg_clean,
        )

        # 4. Product Cards Decision:
        # ONLY return product cards for active recommendations / catalog searches!
        # Do NOT return product cards for direct questions about discounts, specs, or policies!
        show_cards = intent in ("recommend",) and len(matching_products) > 0

        # Provenance metadata
        sources = [
            {
                "title": c.get("document_title", "Policy"),
                "section": c.get("section_title", "General"),
                "score": c.get("score", 0.0),
            }
            for c in policy_chunks
        ]

        return {
            "reply": llm_reply,
            "intent": intent,
            "products": matching_products if show_cards else [],
            "matching_products": matching_products if show_cards else [],
            "sources": sources,
            "product_count": len(matching_products),
            "status": "success",
        }

    # ── Intent Classifier ─────────────────────────────────────────────────────

    def _classify_intent(
        self,
        message: str,
        history: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """
        Classifies user query intent using boundary matching and heuristic rules.
        """
        msg_low = message.lower()

        def has(*words: str) -> bool:
            return any(re.search(rf"\b{re.escape(w)}\b", msg_low) for w in words)

        # Order tracking pattern: TM-12345
        if re.search(r"\bTM-\d{4,8}\b", message, re.I) or has("track", "tracking", "order status"):
            return {"intent": "order_lookup", "product": ""}

        # Refund or cancellation
        if has("refund", "refunds", "cancel order"):
            return {"intent": "refund", "product": ""}

        # Policy questions
        if has("return", "returns", "exchange", "warranty", "shipping", "delivery", "support hours", "contact"):
            # Check if user is asking about return/warranty of a specific product -> mixed
            for prod in self.repo.get_all_names_sample():
                if prod.lower() in msg_low:
                    return {"intent": "mixed", "product": prod}
            return {"intent": "policy", "product": ""}

        # Specific product inquiries: discount, price, specs, stock
        # Must be checked BEFORE recommend/suggest to avoid "sticky product card" trap!
        if (
            has("discount", "discounts", "price", "cost", "how much", "warranty", "stock", "spec", "specs")
            or any(phrase in msg_low for phrase in ["tell me about", "do you sell", "any more discount", "cheaper"])
        ):
            # Extract product name if mentioned
            product_name = self._extract_product_mention(message)
            return {"intent": "product_question", "product": product_name}

        # Catalog browsing / Recommendation requests
        if has("recommend", "suggest", "options", "alternatives", "show me", "browse", "list", "products"):
            product_name = self._extract_product_mention(message)
            return {"intent": "recommend", "product": product_name}

        # Default: product question if a product is mentioned, else general
        product_name = self._extract_product_mention(message)
        if product_name:
            return {"intent": "product_question", "product": product_name}

        return {"intent": "general", "product": ""}

    def _extract_product_mention(self, message: str) -> str:
        """Find if a catalog product name is in the query."""
        all_prods = self.repo.search(limit=50)
        msg_low = message.lower()
        for p in all_prods:
            if p.name.lower() in msg_low or p.brand.lower() in msg_low:
                return p.name
        return ""

    # ── Prompt Builder ────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        intent: str,
        policy_chunks: List[Dict[str, Any]],
        products: List[Dict[str, Any]],
    ) -> str:
        instructions = [
            "You are Clara, TechMart's intelligent and helpful AI assistant.",
            "Always be concise, professional, warm, and strictly factual.",
            "CRITICAL RULES:",
            "1. NEVER invent product prices, discounts, or stock numbers. Use ONLY the verified data provided below.",
            "2. If the user asks about discounts or pricing for a product, state the exact discount and price directly. Do NOT repeat long product lists.",
            "3. If a policy is cited, use the exact rules from the verified policy documents below.",
            "4. If information is not in the provided context, politely say you don't have that detail.",
        ]

        # Context sections
        context_parts = []

        if products:
            context_parts.append("\n=== VERIFIED PRODUCT CATALOG DATA ===")
            for p in products:
                disc_text = f"{p['discount']}% off" if p.get("discount") else "No discount"
                stock_text = f"{p['stock']} in stock" if p.get("stock", 0) > 0 else "Out of stock"
                context_parts.append(
                    f"- {p['name']} (SKU: {p['sku']}): "
                    f"Price: ${p['price']} | Effective Price: ${p.get('effective_price', p['price'])} | "
                    f"Discount: {disc_text} | Stock: {stock_text} | Brand: {p['brand']} | Category: {p['category']}"
                )

        if policy_chunks:
            context_parts.append("\n=== VERIFIED COMPANY POLICY DOCUMENTS ===")
            for c in policy_chunks:
                context_parts.append(f"[{c.get('section_title') or c.get('document_title')}]:\n{c['text']}")

        return "\n".join(instructions) + "\n" + "\n".join(context_parts)
