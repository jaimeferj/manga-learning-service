from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class OcrLine(BaseModel):
    text: str
    tightBoundingBox: BoundingBox
    forcedOrientation: Literal["horizontal", "vertical"]
    isMerged: bool = False

    model_config = ConfigDict(extra="allow")


class OcrPageRequest(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None
    language: str = "ja"
    manga_id: str | None = None
    chapter_id: str | None = None
    page_index: int | None = None
    force: bool = False
    persist: bool = False


class OcrRegionRequest(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None
    region: BoundingBox
    language: str = "ja"


class OcrPageResult(BaseModel):
    img_width: int
    img_height: int
    lines: list[OcrLine]
    cached: bool = False
    backend: str = "mokuro"


class OcrRegionResult(BaseModel):
    text: str
    tightBoundingBox: BoundingBox
    forcedOrientation: Literal["horizontal", "vertical"] = "horizontal"
    isMerged: bool = False


class OcrStatus(BaseModel):
    backend: str
    ready: bool
    error: str | None = None
    cache_entries: int = 0


class OcrPurgeRequest(BaseModel):
    manga_id: str | None = None
    chapter_id: str | None = None


class OcrPurgeResponse(BaseModel):
    removed: int


ImageBase64 = Annotated[str, Field(min_length=1)]
