from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnkiCreateCardRequest(BaseModel):
    deck: str
    model: str
    fields: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    picture_base64: str | None = None
    picture_filename: str | None = None
    picture_field: str | None = None


class AnkiCreateCardResponse(BaseModel):
    note_id: int
    deck: str
    model: str


class AnkiStatus(BaseModel):
    enabled: bool
    reachable: bool
    version: int | None = None
    error: str | None = None


Direction = Literal["explain", "fields"]
