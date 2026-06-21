from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from manga_learning_service.ai.providers import build_provider
from manga_learning_service.ai.types import (
    SYSTEM_CARD_FIELDS,
    SYSTEM_EXPLAIN,
    AiCardFieldsRequest,
    AiCardFieldsResponse,
    AiExplainRequest,
    AiExplainResponse,
    parse_json_response,
)
from manga_learning_service.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/status")
async def ai_status() -> dict[str, str]:
    settings = get_settings()
    return {
        "provider": settings.ai_provider,
        "enabled": "true" if settings.ai_enabled else "false",
    }


@router.post("/explain", response_model=AiExplainResponse)
async def explain(payload: AiExplainRequest) -> AiExplainResponse:
    settings = get_settings()
    if not settings.ai_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "AI disabled (set MANGA_LEARNING_AI_ENABLED=true)",
        )
    provider = build_provider(settings)
    user_prompt = (
        f"Language: {payload.language}\n"
        f"Level: {payload.level}\n"
        f"Text:\n{payload.text}\n"
    )
    raw = await provider.complete(system=SYSTEM_EXPLAIN, user=user_prompt)
    try:
        data = parse_json_response(provider.name, raw)
    except ValueError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    vocab = [
        {"term": str(item.get("term", "")), "reading": str(item.get("reading", "")), "meaning": str(item.get("meaning", ""))}
        for item in data.get("vocab", [])
        if isinstance(item, dict)
    ]
    grammar = [str(item) for item in data.get("grammar", []) if item]
    return AiExplainResponse(
        provider=provider.name,
        explanation=str(data.get("explanation", "")),
        vocab=vocab,
        grammar=grammar,
    )


@router.post("/card-fields", response_model=AiCardFieldsResponse)
async def card_fields(payload: AiCardFieldsRequest) -> AiCardFieldsResponse:
    settings = get_settings()
    if not settings.ai_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "AI disabled (set MANGA_LEARNING_AI_ENABLED=true)",
        )
    provider = build_provider(settings)
    user_prompt = (
        f"Language: {payload.language}\n"
        f"Model hint: {payload.model or 'default'}\n"
        f"Text:\n{payload.text}\n"
    )
    if payload.reading:
        user_prompt += f"\nKnown reading:\n{payload.reading}\n"
    raw = await provider.complete(system=SYSTEM_CARD_FIELDS, user=user_prompt)
    try:
        fields = parse_json_response(provider.name, raw)
    except ValueError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    if not isinstance(fields, dict):
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "AI provider did not return a JSON object",
        )
    coerced = {str(k): str(v) for k, v in fields.items() if v is not None}
    return AiCardFieldsResponse(provider=provider.name, fields=coerced)
