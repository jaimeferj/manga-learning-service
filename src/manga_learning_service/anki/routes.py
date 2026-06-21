from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, HTTPException, Request, status

from manga_learning_service.anki.client import AnkiConnect, AnkiError
from manga_learning_service.anki.types import (
    AnkiCreateCardRequest,
    AnkiCreateCardResponse,
    AnkiStatus,
)
from manga_learning_service.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


async def _connect() -> AnkiConnect:
    settings = get_settings()
    return AnkiConnect(settings.anki_connect_url)


@router.get("/status", response_model=AnkiStatus)
async def anki_status() -> AnkiStatus:
    settings = get_settings()
    status_model = AnkiStatus(enabled=settings.anki_enabled, reachable=False)
    if not settings.anki_enabled:
        return status_model
    try:
        client = await _connect()
        status_model.version = await client.version()
        status_model.reachable = True
    except Exception as exc:
        status_model.error = str(exc)
    return status_model


@router.post("/create-card", response_model=AnkiCreateCardResponse)
async def create_card(payload: AnkiCreateCardRequest, request: Request) -> AnkiCreateCardResponse:
    settings = get_settings()
    if not settings.anki_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "anki integration disabled (set MANGA_LEARNING_ANKI_ENABLED=true)",
        )

    client = await _connect()
    picture = None
    if payload.picture_base64 and payload.picture_filename and payload.picture_field:
        picture = {
            "filename": payload.picture_filename,
            "data": base64.b64decode(payload.picture_base64),
            "fields": [payload.picture_field],
        }
    try:
        note_id = await client.add_note(
            deck=payload.deck,
            model=payload.model,
            fields=payload.fields,
            tags=payload.tags,
            picture=picture,
        )
    except AnkiError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"AnkiConnect error: {exc}") from exc
    return AnkiCreateCardResponse(note_id=note_id, deck=payload.deck, model=payload.model)
