"""
app/config.py
─────────────────────────────────────────────────────────────────────────────
Central configuration loaded from environment variables / .env file.
All secrets must be set in .env or the host environment.  Nothing is
hard-coded here — defaults are intentionally set to safe/empty values.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "TechMart Enterprise RAG"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    ADMIN_API_KEY: str = "change_this_in_production"

    # ── Database ──────────────────────────────────────────────────────────────
    # On Railway: set DATABASE_URL to postgres:// connection string (recommended)
    # or leave blank to use SQLite on the mounted volume at RAILWAY_VOLUME_MOUNT_PATH
    DATABASE_URL: str = "sqlite:///./data/techmart_local.db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    # SQLite fallback for local dev without Postgres
    SQLITE_FALLBACK: bool = True
    SQLITE_DB_PATH: str = "./data/techmart_local.db"

    # Railway persistent volume mount path (set automatically by Railway when a volume is attached)
    # When set, SQLite and ChromaDB paths are relocated under this directory.
    RAILWAY_VOLUME_MOUNT_PATH: str = ""

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "company_policies"

    # ── Embedding Provider ────────────────────────────────────────────────────
    # Supported: "openai" | "gemini" | "huggingface" | "chroma_default"
    EMBEDDING_PROVIDER: str = "chroma_default"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # Supported: "groq" | "openai" | "gemini"
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-3.5-flash-lite"
    GROQ_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048

    # ── Spreadsheet Sync ──────────────────────────────────────────────────────
    SYNC_SOURCE_TYPE: str = "google_sheets"   # "google_sheets" | "csv" | "excel"
    GOOGLE_SHEETS_SPREADSHEET_ID: str = "15wQuB7ZcJBuEfODW82JyXDz9tM1XhB33DXl-nKDi2Z4"
    GOOGLE_SHEETS_SHEET_NAME: str = "Product Details"
    GOOGLE_SERVICE_ACCOUNT_JSON_PATH: str = "./techmart-database-a6718d22604c.json"
    GOOGLE_SERVICE_ACCOUNT_JSON: str = ""    # Overrides file if non-empty
    LOCAL_CSV_BACKUP_PATH: str = "./data/sample_products.csv"
    SYNC_POLL_INTERVAL_SECONDS: int = 30

    # ── Policy Chunking ───────────────────────────────────────────────────────
    CHUNK_SIZE_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 75
    # Minimum chunk size in tokens (smaller chunks are merged with next)
    CHUNK_MIN_TOKENS: int = 50
    # Max chunks returned per policy retrieval call
    POLICY_TOP_K: int = 5
    # Similarity threshold — chunks below this score are discarded
    POLICY_SCORE_THRESHOLD: float = 0.35

    # ── Product retrieval ─────────────────────────────────────────────────────
    PRODUCT_TOP_K: int = 10

    # ── Rate limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton — import this everywhere
settings = Settings()
