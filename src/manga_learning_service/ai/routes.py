from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from manga_learning_service.ai.providers import build_provider
from manga_learning_service.ai.types import (
    ACTION_PROMPTS,
    SYSTEM_CARD_FIELDS,
    SYSTEM_EXPLAIN,
    SYSTEM_LEARN,
    AiCardFieldsRequest,
    AiCardFieldsResponse,
    AiExplainRequest,
    AiExplainResponse,
    AiLearningSection,
    AiLearnRequest,
    AiLearnResponse,
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


@router.post("/learn", response_model=AiLearnResponse)
async def learn(payload: AiLearnRequest) -> AiLearnResponse:
    settings = get_settings()
    if not settings.ai_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "AI disabled (set MANGA_LEARNING_AI_ENABLED=true)",
        )
    provider = build_provider(settings)
    context_lines = [
        f"Previous line: 「{payload.context.previous_line}」" if payload.context.previous_line else None,
        f"Next line: 「{payload.context.next_line}」" if payload.context.next_line else None,
        f"Speaker: {payload.context.speaker}" if payload.context.speaker else None,
        f"Scene/context: {payload.context.scene}" if payload.context.scene else None,
    ]
    context = "\n".join(line for line in context_lines if line) or "No surrounding context supplied."
    user_prompt = (
        f"Task: {ACTION_PROMPTS[payload.action]}\n\n"
        f"Manga sentence:\n「{payload.sentence}」\n\n"
        f"Optional surrounding context:\n{context}\n\n"
        f"User level:\nSpanish-speaking Japanese learner, {payload.level}."
    )
    raw = await provider.complete(system=SYSTEM_LEARN, user=user_prompt, max_tokens=1200)
    try:
        data = parse_json_response(provider.name, raw)
        raw_sections = data.get("sections", [])
        sections = [
            AiLearningSection(label=str(item["label"]), content=str(item["content"]))
            for item in raw_sections
            if isinstance(item, dict) and item.get("label") and item.get("content")
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"invalid AI learning response: {exc}") from exc
    if not sections:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "AI provider returned no learning sections")
    return AiLearnResponse(
        provider=provider.name,
        action=payload.action,
        sentence=payload.sentence,
        sections=sections,
    )


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
