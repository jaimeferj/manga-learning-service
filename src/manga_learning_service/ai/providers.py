from __future__ import annotations

import logging

import httpx

from manga_learning_service.ai.provider import AiProvider
from manga_learning_service.config import Settings

logger = logging.getLogger(__name__)


def _default_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


class OpenAiProvider(AiProvider):
    name = "openai"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> str:
        if not self._settings.ai_openai_api_key:
            raise RuntimeError("MANGA_LEARNING_AI_OPENAI_API_KEY not set")
        headers = {
            "Authorization": f"Bearer {self._settings.ai_openai_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._settings.ai_openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        client = self._client or _default_client(30.0)
        owns_client = self._client is None
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=body,
            )
        finally:
            if owns_client:
                await client.aclose()
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("openai returned no choices")
        return str(choices[0]["message"]["content"]).strip()


class OllamaProvider(AiProvider):
    name = "ollama"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> str:
        url = self._settings.ai_ollama_url.rstrip("/") + "/api/chat"
        body = {
            "model": self._settings.ai_ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        client = self._client or _default_client(60.0)
        owns_client = self._client is None
        try:
            response = await client.post(url, json=body)
        finally:
            if owns_client:
                await client.aclose()
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("ollama returned no message content")
        return content.strip()


def build_provider(settings: Settings) -> AiProvider:
    if settings.ai_provider == "openai":
        return OpenAiProvider(settings)
    if settings.ai_provider == "ollama":
        return OllamaProvider(settings)
    raise ValueError(f"unknown AI provider: {settings.ai_provider}")
