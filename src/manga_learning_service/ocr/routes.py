from __future__ import annotations

import asyncio
import io as _io
import logging
from typing import Any, cast

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from PIL import Image as PILImage

from manga_learning_service.cache.keys import PageIdentity, hash_bytes
from manga_learning_service.config import get_settings
from manga_learning_service.ocr.service import OcrEngine, decode_image_async, engine_factory
from manga_learning_service.ocr.types import (
    OcrLine,
    OcrPageRequest,
    OcrPageResult,
    OcrPurgeRequest,
    OcrPurgeResponse,
    OcrRegionRequest,
    OcrRegionResult,
    OcrStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter()
_engine_lock = asyncio.Lock()
_engine: OcrEngine | None = None
_http_client: httpx.AsyncClient | None = None
_http_client_lock = asyncio.Lock()


async def get_engine() -> OcrEngine:
    global _engine
    async with _engine_lock:
        if _engine is None:
            settings = get_settings()
            _engine = engine_factory(settings.ocr_backend, force_cpu=settings.ocr_force_cpu)
            await asyncio.to_thread(_engine.load)
    return _engine


async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    async with _http_client_lock:
        if _http_client is None:
            _http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        return _http_client


async def close_http_client() -> None:
    global _http_client
    async with _http_client_lock:
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def map_mokuro_blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    img_w = float(result.get("img_width", 0)) or 1.0
    img_h = float(result.get("img_height", 0)) or 1.0
    for blk in result.get("blocks", []):
        box = blk.get("box")
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        tight = {
            "x": max(0.0, min(1.0, float(x0) / img_w)),
            "y": max(0.0, min(1.0, float(y0) / img_h)),
            "width": max(0.0, min(1.0, (float(x1) - float(x0)) / img_w)),
            "height": max(0.0, min(1.0, (float(y1) - float(y0)) / img_h)),
        }
        orientation = "vertical" if bool(blk.get("vertical", False)) else "horizontal"
        text = " ".join(str(t) for t in blk.get("lines", [])).strip()
        if not text:
            continue
        lines.append(
            {
                "text": text,
                "tightBoundingBox": tight,
                "forcedOrientation": orientation,
                "isMerged": False,
            }
        )
    return lines


@router.get("/status", response_model=OcrStatus)
async def ocr_status(request: Request) -> OcrStatus:
    engine = await get_engine()
    cache = request.app.state.cache
    settings = get_settings()
    return OcrStatus(
        backend=settings.ocr_backend,
        ready=engine.ready,
        error=engine.error,
        cache_entries=await cache.count(),
    )


@router.post("/page", response_model=OcrPageResult)
async def ocr_page(payload: OcrPageRequest, request: Request) -> OcrPageResult:
    settings = get_settings()
    cache = request.app.state.cache
    engine = await get_engine()

    try:
        http_client = await get_http_client()
        image_bytes = await decode_image_async(
            payload.image_base64,
            payload.image_url,
            http_client=http_client,
            manga_server_url=settings.manga_server_url,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"failed to fetch image_url: {exc}") from exc

    if len(image_bytes) > settings.ocr_max_image_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "image too large")

    identity = PageIdentity(
        manga_id=payload.manga_id,
        chapter_id=payload.chapter_id,
        page_index=payload.page_index,
        image_url=payload.image_url,
        image_hash=hash_bytes(image_bytes),
        language=payload.language,
        backend=settings.ocr_backend,
    )

    if not payload.force:
        cached = await cache.get_page(identity)
        if cached is not None:
            return OcrPageResult(
                img_width=int(cached["img_width"]),
                img_height=int(cached["img_height"]),
                lines=[OcrLine.model_validate(line) for line in cached["lines"]],
                cached=True,
                backend=settings.ocr_backend,
            )

    if not engine.ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            engine.error or "ocr engine not ready",
        )

    raw = await engine.recognize_page(image_bytes)
    lines = cast(list[OcrLine], [OcrLine.model_validate(line) for line in map_mokuro_blocks(raw)])
    img_width = int(raw.get("img_width", 0))
    img_height = int(raw.get("img_height", 0))
    await cache.put_page(
        identity,
        {
            "img_width": img_width,
            "img_height": img_height,
            "lines": [line.model_dump() for line in lines],
        },
    )

    return OcrPageResult(
        img_width=img_width,
        img_height=img_height,
        lines=lines,
        cached=False,
        backend=settings.ocr_backend,
    )


@router.post("/recognize-region", response_model=OcrRegionResult)
async def ocr_region(payload: OcrRegionRequest) -> OcrRegionResult:
    engine = await get_engine()
    settings = get_settings()
    try:
        http_client = await get_http_client()
        image_bytes = await decode_image_async(
            payload.image_base64,
            payload.image_url,
            http_client=http_client,
            manga_server_url=settings.manga_server_url,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"failed to fetch image_url: {exc}") from exc

    if not engine.ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            engine.error or "ocr engine not ready",
        )

    full = PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
    w, h = full.size
    region = payload.region
    box = (
        int(region.x * w),
        int(region.y * h),
        int((region.x + region.width) * w),
        int((region.y + region.height) * h),
    )
    cropped = full.crop(box)
    buf = _io.BytesIO()
    cropped.save(buf, format="PNG")
    text = await engine.recognize_text(buf.getvalue())
    return OcrRegionResult(
        text=text.strip(),
        tightBoundingBox=region,
        forcedOrientation="horizontal",
        isMerged=False,
    )


@router.post("/purge-cache", response_model=OcrPurgeResponse)
async def purge_cache(payload: OcrPurgeRequest, request: Request) -> OcrPurgeResponse:
    cache = request.app.state.cache
    removed = await cache.purge(payload.manga_id, payload.chapter_id)
    return OcrPurgeResponse(removed=removed)
