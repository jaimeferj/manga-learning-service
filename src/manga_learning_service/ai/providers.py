from __future__ import annotations

import logging

import httpx

from manga_learning_service.ai.provider import AiProvider
from manga_learning_service.config import Settings

logger = logging.getLogger(__name__)


def _default_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


class _OpenAiCompatibleProvider(AiProvider):
    name: str

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        api_key_env: str,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.client = client

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(f"{self.name} API key not set ({self.api_key_env})")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if self.name == "minimax":
            body["max_completion_tokens"] = max_tokens
            body["reasoning_split"] = True
        else:
            body["max_tokens"] = max_tokens
        client = self.client or _default_client(self.timeout_seconds)
        owns_client = self.client is None
        try:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
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
            raise RuntimeError(f"{self.name} returned no choices")
        return str(choices[0]["message"]["content"]).strip()


class MinimaxProvider(AiProvider):
    name = "minimax"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        self.timeout_seconds = settings.ai_minimax_timeout_seconds

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> str:
        api_key = self._settings.ai_minimax_api_key
        if not api_key:
            raise RuntimeError("minimax API key not set (MANGA_LEARNING_AI_MINIMAX_API_KEY)")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, object] = {
            "model": self._settings.ai_minimax_model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._settings.ai_minimax_thinking:
            body["thinking"] = {"type": "adaptive"}
        url = self._settings.ai_minimax_base_url.rstrip("/") + "/v1/messages"
        client = self._client or _default_client(self.timeout_seconds)
        owns_client = self._client is None
        try:
            response = await client.post(url, headers=headers, json=body)
        finally:
            if owns_client:
                await client.aclose()
        response.raise_for_status()
        data = response.json()
        parts = data.get("content") or []
        text_parts = [
            str(part.get("text", ""))
            for part in parts
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        if not text_parts:
            raise RuntimeError("minimax returned no text content")
        return "\n".join(text_parts).strip()


class OpenAiProvider(_OpenAiCompatibleProvider):
    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(
            name="openai",
            base_url=settings.ai_openai_base_url,
            api_key=settings.ai_openai_api_key,
            model=settings.ai_openai_model,
            api_key_env="MANGA_LEARNING_AI_OPENAI_API_KEY",
            client=client,
        )


class OllamaProvider(AiProvider):
    name = "ollama"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
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
    if settings.ai_provider == "minimax":
        return MinimaxProvider(settings)
    if settings.ai_provider == "openai":
        return OpenAiProvider(settings)
    if settings.ai_provider == "ollama":
        return OllamaProvider(settings)
    raise ValueError(f"unknown AI provider: {settings.ai_provider}")
