"""
app/services/llm_service.py
─────────────────────────────────────────────────────────────────────────────
Unified LLM Client supporting Groq (primary) and Google Gemini (fallback).

Features:
- Strip thinking tags (<think>...</think>) and raw XML brackets.
- Strict temperature=0.0 default for factual deterministic grounding.
- Automatic provider failover: Groq -> Gemini.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
# Preserve functional UI tags: CART_BUTTON and ORDER_FORM
_DISALLOWED_TAG_RE = re.compile(r"\[\[(?!CART_BUTTON|ORDER_FORM)[A-Z_]+:.*?\]\]", re.S)


def clean_llm_text(text: str) -> str:
    """Strip internal reasoning blocks and unauthorized tags, while preserving UI interactive action tags."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    cleaned = _DISALLOWED_TAG_RE.sub("", cleaned)
    return cleaned.strip()


class LLMService:
    def __init__(self):
        self.groq_api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        self.gemini_api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate response with primary provider and automatic fallback."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # If primary provider is Gemini
        if settings.LLM_PROVIDER == "gemini" and self.gemini_api_key:
            try:
                resp = self._call_gemini(system_prompt, user_message, temp, tokens, history)
                if resp:
                    return clean_llm_text(resp)
            except Exception as exc:
                logger.warning("Gemini call failed (%s). Attempting Groq fallback.", exc)

        # Attempt Groq
        if self.groq_api_key:
            try:
                resp = self._call_groq(system_prompt, user_message, temp, tokens)
                if resp:
                    return clean_llm_text(resp)
            except Exception as exc:
                logger.warning("Groq call failed (%s). Attempting Gemini fallback.", exc)

        # Fallback to Gemini if not already attempted
        if settings.LLM_PROVIDER != "gemini" and self.gemini_api_key:
            try:
                resp = self._call_gemini(system_prompt, user_message, temp, tokens, history)
                if resp:
                    return clean_llm_text(resp)
            except Exception as exc:
                logger.error("Gemini call also failed: %s", exc)

        return (
            "I'm sorry, I'm currently unable to reach my knowledge service. "
            "Please try again in a few moments or contact support@techmart.com."
        )

    def _call_groq(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]

    def _call_gemini(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-3.5-flash-lite:generateContent?key={self.gemini_api_key}"
        )
        headers = {"Content-Type": "application/json"}

        # Build multi-turn contents array for proper conversation context
        contents = []

        # Inject prior turns so Gemini has full conversation memory
        for turn in (history or []):
            g_role = "user" if turn.get("role") == "user" else "model"
            content_text = turn.get("content", "")
            if content_text:
                contents.append({"role": g_role, "parts": [{"text": content_text}]})

        # Final user turn includes the system prompt + current question
        final_user_text = f"{system_prompt}\n\nUser Question: {user_message}"
        contents.append({"role": "user", "parts": [{"text": final_user_text}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return ""


# Singleton instance
llm_service = LLMService()
