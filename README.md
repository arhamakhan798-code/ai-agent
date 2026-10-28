# TechMart AI Enterprise RAG Chatbot (Clara)

A modular, production-ready RAG application combining:
- **Relational Product Catalog**: PostgreSQL / SQLite for 1,000+ products with strict pricing & inventory integrity.
- **ChromaDB Policy Retrieval**: Structure-aware semantic search over internal company policies (returns, warranties, shipping, customer service) with provenance citations.
- **Dynamic Intent Routing**: Intelligent classification that answers follow-up product questions (discounts, specs, stock) directly without getting trapped in repetitive product cards.
- **Multi-Format Document Ingestion**: Unified loading and chunking for `.txt`, `.md`, `.pdf` (with OCR detection), and `.docx` with SHA-256 deduplication.

---

## 🏗️ Architecture Overview

```
├── app/
│   ├── config.py                 # Central Pydantic BaseSettings
│   ├── main.py                   # FastAPI application & lifespan management
│   ├── core/
│   │   └── database.py           # SQLAlchemy engine & session factory (PostgreSQL / SQLite)
│   ├── models/
│   │   ├── product.py            # Product model with calculated effective_price
│   │   ├── sync_state.py         # Sync audit state & content hashing
│   │   └── order.py              # Order & refund models with timestamps
│   ├── repositories/
│   │   └── product_repository.py # Fast relational querying, indexing, and batch upsert
│   ├── schemas/
│   │   ├── product.py            # Pydantic validation for products and filters
│   │   └── sync.py               # Pydantic schemas for sync triggers & status
│   ├── services/
│   │   ├── document_loader.py    # Multi-format doc parser (.txt, .md, .pdf, .docx)
│   │   ├── chunking.py           # Recursive structure-aware token chunker (tiktoken)
│   │   ├── embeddings.py         # Pluggable embedding providers (ChromaDB MiniLM, OpenAI, Gemini)
│   │   ├── chroma_service.py     # ChromaDB PersistentClient vector store wrapper
│   │   ├── policy_ingestion.py   # Ingestion pipeline with file-hash deduplication
│   │   ├── product_sync.py       # Google Sheets & CSV sync service
│   │   ├── llm_service.py        # Groq (primary) + Gemini (fallback) client
│   │   └── rag_engine.py         # Hybrid RAG router, prompt assembly, and cards logic
│   └── api/
│       └── v1/
│           ├── chat.py           # Chat & RAG answer endpoints
│           ├── products.py       # Product catalog REST endpoints
│           ├── sync.py           # Catalog synchronization endpoints
│           └── orders.py         # Order placement, lookup, and refund endpoints
├── data/
│   ├── policies/                 # Policy documents & metadata manifest
│   └── sample_products.csv       # Fallback sample product catalog
├── scripts/
│   ├── ingest_policies.py        # CLI for batch/single policy ingestion
│   ├── sync_products.py          # CLI for Google Sheets / CSV sync
│   └── evaluate_retrieval.py     # Retrieval benchmark & evaluation metrics
├── tests/
│   ├── test_chunking.py          # Unit tests for chunking & document loaders
│   ├── test_ingestion.py         # Integration tests for ChromaDB policy pipeline
│   ├── test_products.py          # Unit tests for relational models & repository
│   └── test_api.py               # End-to-end FastAPI API integration tests
└── widget.html                   # Luxury interactive customer chat interface
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
```bash
cp .env.example .env
```
Ensure `GROQ_API_KEY` or `GEMINI_API_KEY` is set.

### 3. Ingest Policy Documents
```bash
# Preview policy chunks
python -m scripts.ingest_policies --preview --file data/policies/policy.txt

# Ingest all policy documents in data/policies/
python -m scripts.ingest_policies
```

### 4. Sync Products Catalog
```bash
# Sync from live Google Sheets (or fallback CSV)
python -m scripts.sync_products --source csv

# Check catalog sync status
python -m scripts.sync_products --status
```

### 5. Run the Server
```bash
# Using uvicorn directly
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload

# Or double-click start_agent.bat on Windows
```

- **Interactive API Docs (Swagger)**: http://localhost:5000/docs
- **Client Web Widget**: Open `widget.html` in your browser or visit http://localhost:5000/

---

## 🧪 Running Automated Tests

Run the complete 89-test verification suite:
```bash
pytest tests/ -v
```

All 89 unit and integration tests verify:
- ✅ Structure-aware token chunking and stable SHA-256 IDs
- ✅ ChromaDB policy vector storage and similarity search
- ✅ Relational product queries, effective price discounts, and batch upsert
- ✅ Google Sheets / CSV sync with content-hash change detection
- ✅ Resolution of the "sticky product cards" bug: follow-up discount inquiries answer directly without repeating cards

---

## 📄 License
MIT License
