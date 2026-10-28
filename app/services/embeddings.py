"""
app/services/embeddings.py
─────────────────────────────────────────────────────────────────────────────
Pluggable embedding layer.

Selecting a provider
────────────────────
Set EMBEDDING_PROVIDER in .env to one of:

    chroma_default   — ChromaDB's built-in all-MiniLM-L6-v2  (no extra key)
    openai           — OpenAI text-embedding-3-small / text-embedding-ada-002
    gemini           — Google Generative AI embedding-001
    huggingface      — Any sentence-transformers model (local inference)

The EmbeddingProvider protocol defines the single method required:
    embed_texts(texts: list[str]) -> list[list[float]]

ChromaDB embedding functions are the primary integration point — the
chromadb.utils.embedding_functions wrappers are returned directly so they
can be passed to chroma_service.py without any adapters.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Protocol (structural typing) ──────────────────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


# ── ChromaDB embedding functions (primary integration) ────────────────────────

def get_chroma_embedding_function(provider: str, model: str, **kwargs):
    """
    Return a ChromaDB-compatible embedding function.
    This is passed directly to chromadb.Collection.

    Parameters
    ──────────
    provider  : 'chroma_default' | 'openai' | 'gemini' | 'huggingface'
    model     : provider-specific model name
    **kwargs  : api_key, etc.
    """
    from chromadb.utils import embedding_functions as ef

    if provider == "chroma_default":
        logger.info("Using ChromaDB default embedding function (all-MiniLM-L6-v2)")
        return ef.DefaultEmbeddingFunction()

    if provider == "openai":
        api_key = kwargs.get("api_key") or ""
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set for embedding_provider=openai")
        logger.info("Using OpenAI embedding function: %s", model)
        return ef.OpenAIEmbeddingFunction(api_key=api_key, model_name=model)

    if provider == "gemini":
        api_key = kwargs.get("api_key") or ""
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set for embedding_provider=gemini")
        logger.info("Using Google Generative AI embedding function")
        try:
            return ef.GoogleGenerativeAiEmbeddingFunction(
                api_key=api_key, model_name=f"models/{model}"
            )
        except AttributeError:
            # Older chromadb version
            return ef.DefaultEmbeddingFunction()

    if provider == "huggingface":
        logger.info("Using HuggingFace SentenceTransformer embedding: %s", model)
        return ef.SentenceTransformerEmbeddingFunction(model_name=model)

    raise ValueError(
        f"Unknown embedding provider: '{provider}'. "
        "Choose from: chroma_default, openai, gemini, huggingface"
    )


# ── Standalone embedder (for non-ChromaDB usage, e.g. product search) ─────────

class StandaloneEmbedder:
    """
    A simple wrapper for generating embeddings outside of ChromaDB.
    Uses the same provider configuration.
    """

    def __init__(self, chroma_ef) -> None:
        self._ef = chroma_ef

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._ef(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedding_function():
    """
    Build and return a ChromaDB embedding function from application settings.
    Call once at startup and reuse the same instance.
    """
    from app.config import settings

    return get_chroma_embedding_function(
        provider=settings.EMBEDDING_PROVIDER,
        model=settings.EMBEDDING_MODEL,
        api_key=(
            settings.OPENAI_API_KEY
            if settings.EMBEDDING_PROVIDER == "openai"
            else settings.GEMINI_API_KEY
        ),
    )
