from __future__ import annotations

import json

import httpx
import pytest

from manga_learning_service.ai.providers import (
    MinimaxProvider,
    OllamaProvider,
    OpenAiProvider,
    build_provider,
)
from manga_learning_service.ai.types import AiLearningPayload, parse_json_response
from manga_learning_service.config import Settings


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, response: dict | list | str) -> None:
        self.response = response
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if isinstance(self.response, (dict, list)):
            return httpx.Response(200, json=self.response)
        return httpx.Response(200, text=self.response)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        ai_provider="minimax",
        ai_minimax_api_key="test-minimax",
        ai_minimax_base_url="https://api.minimax.io/anthropic",
        ai_minimax_model="MiniMax-M3",
        ai_openai_api_key="test-openai",
        ai_openai_model="gpt-4o-mini",
        ai_openai_base_url="https://api.openai.com/v1",
        ai_ollama_url="http://ollama.test",
        ai_ollama_model="llama3.1",
    )


async def test_minimax_provider_returns_content(settings: Settings) -> None:
    transport = _Transport(
        {
            "content": [
                {"type": "thinking", "thinking": "..."},
                {"type": "text", "text": '{"explanation": "ok"}'},
            ]
        }
    )
    http = httpx.AsyncClient(transport=transport)
    provider = MinimaxProvider(settings, client=http)
    try:
        result = await provider.complete(system="s", user="u")
    finally:
        await http.aclose()
    assert result == '{"explanation": "ok"}'
    request = transport.calls[0]
    assert request.headers["authorization"] == "Bearer test-minimax"
    body = json.loads(request.read().decode("utf-8"))
    assert body["model"] == "MiniMax-M3"
    assert body["system"] == "s"
    assert body["max_tokens"] == 600
    assert "thinking" not in body
    assert "max_completion_tokens" not in body
    assert "reasoning_split" not in body
    assert str(request.url) == "https://api.minimax.io/anthropic/v1/messages"
    assert provider.timeout_seconds == 90.0


async def test_minimax_provider_thinking_enabled(settings: Settings) -> None:
    transport = _Transport(
        {"content": [{"type": "text", "text": "ok"}]}
    )
    http = httpx.AsyncClient(transport=transport)
    settings.ai_minimax_thinking = True
    provider = MinimaxProvider(settings, client=http)
    try:
        await provider.complete(system="s", user="u")
    finally:
        await http.aclose()
    body = json.loads(transport.calls[0].read().decode("utf-8"))
    assert body["thinking"] == {"type": "adaptive"}


async def test_minimax_provider_raises_without_text_blocks(settings: Settings) -> None:
    transport = _Transport({"content": [{"type": "thinking", "thinking": "..."}]})
    http = httpx.AsyncClient(transport=transport)
    provider = MinimaxProvider(settings, client=http)
    try:
        with pytest.raises(RuntimeError, match="no text content"):
            await provider.complete(system="s", user="u")
    finally:
        await http.aclose()


async def test_openai_provider_returns_content(settings: Settings) -> None:
    transport = _Transport(
        {
            "choices": [
                {"message": {"role": "assistant", "content": '{"explanation": "ok"}'}},
            ]
        }
    )
    http = httpx.AsyncClient(transport=transport)
    provider = OpenAiProvider(settings, client=http)
    try:
        result = await provider.complete(system="s", user="u")
    finally:
        await http.aclose()
    assert result == '{"explanation": "ok"}'
    request = transport.calls[0]
    assert request.headers["authorization"] == "Bearer test-openai"
    body = json.loads(request.read().decode("utf-8"))
    assert body["model"] == "gpt-4o-mini"
    assert body["max_tokens"] == 600
    assert "reasoning_split" not in body


async def test_provider_raises_when_api_key_missing() -> None:
    settings = Settings(ai_minimax_api_key="")
    provider = MinimaxProvider(settings)
    with pytest.raises(RuntimeError, match="MANGA_LEARNING_AI_MINIMAX_API_KEY"):
        await provider.complete(system="s", user="u")


async def test_build_provider_dispatches_to_minimax(settings: Settings) -> None:
    provider = build_provider(settings)
    assert isinstance(provider, MinimaxProvider)


async def test_ollama_provider_returns_content(settings: Settings) -> None:
    transport = _Transport({"message": {"role": "assistant", "content": "hello"}})
    http = httpx.AsyncClient(transport=transport, base_url="http://ollama.test")
    provider = OllamaProvider(settings, client=http)
    try:
        result = await provider.complete(system="s", user="u")
    finally:
        await http.aclose()
    assert result == "hello"


def test_parse_json_response_handles_fences() -> None:
    raw = "```json\n{\"a\": 1}\n```"
    assert parse_json_response("minimax", raw) == {"a": 1}


def test_parse_json_response_handles_reasoning_before_json() -> None:
    raw = '<think>I should answer in Spanish.</think>\n{"sections": [{"label": "Estructura", "content": "ok"}]}'
    assert parse_json_response("minimax", raw) == {
        "sections": [{"label": "Estructura", "content": "ok"}]
    }


def test_parse_json_response_rejects_truncated_outer_object() -> None:
    raw = '{"sections": [{"label": "Estructura", "content": "ok"}, {"label": "Partículas"'
    with pytest.raises(ValueError):
        parse_json_response("minimax", raw)


def test_parse_json_response_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        parse_json_response("minimax", "not json")


def test_learning_payload_rejects_action_specific_shape() -> None:
    with pytest.raises(ValueError):
        AiLearningPayload.model_validate(
            {"items": [{"pattern": "〜ば", "meaning": "si"}]}
        )


def test_learning_payload_accepts_only_label_content_sections() -> None:
    payload = AiLearningPayload.model_validate(
        {"sections": [{"label": "Estructura", "content": "Condicional con ば."}]}
    )
    assert payload.sections[0].label == "Estructura"
