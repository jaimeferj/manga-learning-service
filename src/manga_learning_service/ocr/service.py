from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_URL_SCHEMES = ("http", "https")

# A normal manga page typically has a height/width ratio in this range. These
# are the defaults; override per source via ``ocr_page_ratio_min`` /
# ``ocr_page_ratio_max`` settings (e.g. 1.55-1.58 for One Piece).
DEFAULT_MANGA_PAGE_RATIO_MIN = 1.25
DEFAULT_MANGA_PAGE_RATIO_MAX = 1.85

# Typical chapter length bounds used to constrain the page-count search.
DEFAULT_MIN_PAGES = 8
DEFAULT_MAX_PAGES = 25

# Overlap between virtual pages, as a fraction of the page height.
DEFAULT_VIRTUAL_PAGE_OVERLAP = 0.1

# IoU threshold above which two text boxes from different virtual pages are
# considered duplicates of the same detection.
DEFAULT_IOU_DEDUP_THRESHOLD = 0.5


@dataclass(slots=True)
class OcrBackend:
    name: str
    ready: bool = False
    error: str | None = None


class OcrEngine(Protocol):
    name: str

    @property
    def ready(self) -> bool: ...

    @property
    def error(self) -> str | None: ...

    def load(self) -> None: ...

    async def recognize_page(self, image_bytes: bytes) -> dict[str, Any]: ...

    async def recognize_text(self, image_bytes: bytes) -> str: ...


class MokuroEngine:
    name = "mokuro"

    def __init__(self, *, force_cpu: bool = True) -> None:
        self._force_cpu = force_cpu
        self._mocr: Any = None
        self._lock = threading.Lock()
        self._ready = False
        self._error: str | None = None

    def load(self) -> None:
        try:
            from mokuro import MangaPageOcr

            with self._lock:
                self._mocr = MangaPageOcr(force_cpu=self._force_cpu)
                self._ready = True
            logger.info("mokuro MangaPageOcr ready")
        except Exception as exc:
            self._error = str(exc)
            logger.exception("failed to load mokuro MangaPageOcr")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    async def recognize_page(self, image_bytes: bytes) -> dict[str, Any]:
        if not self._ready:
            raise RuntimeError(self._error or "ocr engine not ready")
        return await asyncio.to_thread(_recognize_stitched_or_virtual, self._mocr, image_bytes)


def estimate_page_count(
    width: int,
    height: int,
    *,
    ratio_min: float = DEFAULT_MANGA_PAGE_RATIO_MIN,
    ratio_max: float = DEFAULT_MANGA_PAGE_RATIO_MAX,
    min_pages: int = DEFAULT_MIN_PAGES,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> int | None:
    """Estimate the number of manga pages in a stitched image.

    Returns the page count whose virtual page aspect ratio is closest to a normal
    manga page, or ``None`` if no candidate yields a sane ratio. A normal manga
    page is roughly ``ratio_min`` to ``ratio_max`` times taller than wide; tune
    these per source (e.g. ``1.55-1.58`` for One Piece). For an image that is
    not a tall strip, returns ``None`` so the caller can fall back to a single
    recognition.
    """
    if width <= 0 or height <= 0:
        return None
    aspect = height / width
    if aspect < 2.5:
        return None
    if ratio_min >= ratio_max:
        return None
    if min_pages > max_pages:
        return None
    best_count: int | None = None
    best_distance = float("inf")
    center = (ratio_min + ratio_max) / 2
    for n in range(min_pages, max_pages + 1):
        page_ratio = aspect / n
        if page_ratio < ratio_min or page_ratio > ratio_max:
            continue
        distance = abs(page_ratio - center)
        if distance < best_distance:
            best_distance = distance
            best_count = n
    return best_count


def _slice_virtual_pages(
    pil_image: Image.Image,
    page_count: int,
    overlap_fraction: float = DEFAULT_VIRTUAL_PAGE_OVERLAP,
) -> list[tuple[bytes, int]]:
    """Slice a tall image into ``page_count`` roughly equal virtual pages with overlap.

    Each slice covers ``height / page_count`` pixels plus an overlap band on the
    bottom so text near a virtual boundary is still detected in at least one of
    the adjacent slices. The last slice is allowed to be shorter to accommodate
    the image's true height. Returns ``(png_bytes, y_offset_in_original)`` tuples.
    """
    width, height = pil_image.size
    base_step = height / page_count
    overlap = int(base_step * overlap_fraction)
    slice_h = int(base_step) + overlap
    out: list[tuple[bytes, int]] = []
    for i in range(page_count):
        top = int(round(i * base_step))
        bottom = min(top + slice_h, height)
        if top >= height:
            break
        cropped = pil_image.crop((0, top, width, bottom))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        out.append((buf.getvalue(), top))
    return out


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _dedupe_blocks(blocks: list[dict[str, Any]], iou_threshold: float = DEFAULT_IOU_DEDUP_THRESHOLD) -> list[dict[str, Any]]:
    """Remove near-duplicate text boxes that come from overlapping virtual pages.

    Keeps the first occurrence of each box and drops later ones whose IoU with any
    kept box is at least ``iou_threshold``. Blocks without a usable ``box`` are
    passed through unchanged.
    """
    kept: list[dict[str, Any]] = []
    kept_boxes: list[tuple[int, int, int, int]] = []
    for block in blocks:
        box = block.get("box")
        if not box or len(box) != 4:
            kept.append(block)
            continue
        key = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        if any(_iou(key, existing) >= iou_threshold for existing in kept_boxes):
            continue
        kept_boxes.append(key)
        kept.append(block)
    return kept


def _recognize_stitched_or_virtual(mocr: Any, image_bytes: bytes) -> dict[str, Any]:
    """Recognize text in an image, splitting tall stitched strips into virtual pages.

    We do not search for whitespace page breaks (those are unreliable: speech
    bubbles, eye whites, and panel backgrounds are all white). Instead, when the
    image aspect ratio looks like a multi-page strip, we estimate the page count
    from a normal manga page proportion and slice the image into that many
    equally-spaced virtual pages with overlap so imperfect boundaries do not lose
    text. Detections from overlapping pages are then deduped by bounding-box IoU.
    """
    from manga_learning_service.config import get_settings

    settings = get_settings()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = pil_image.size
    page_count = estimate_page_count(
        width,
        height,
        ratio_min=settings.ocr_page_ratio_min,
        ratio_max=settings.ocr_page_ratio_max,
        min_pages=settings.ocr_min_pages,
        max_pages=settings.ocr_max_pages,
    )
    if page_count is None:
        return _run_mokuro(mocr, image_bytes)

    logger.info("stitched image detected (%dx%d); using %d virtual pages", width, height, page_count)
    slices = _slice_virtual_pages(pil_image, page_count, overlap_fraction=settings.ocr_virtual_page_overlap)

    merged_blocks: list[dict[str, Any]] = []
    for slice_bytes, y_offset in slices:
        result = _run_mokuro(mocr, slice_bytes)
        for block in result.get("blocks", []):
            box = block.get("box")
            if not box or len(box) != 4:
                continue
            x0, y0, x1, y1 = (int(v) for v in box)
            block_copy = dict(block)
            block_copy["box"] = [x0, y_offset + y0, x1, y_offset + y1]
            merged_blocks.append(block_copy)

    merged_blocks = _dedupe_blocks(merged_blocks, iou_threshold=settings.ocr_iou_dedup_threshold)
    return {
        "version": "0.2.0",
        "img_width": width,
        "img_height": height,
        "blocks": merged_blocks,
    }


def _run_mokuro(mocr: Any, image_bytes: bytes) -> dict[str, Any]:
    """Run mokuro on image bytes via a temp file and return the raw result dict."""
    tmp = Path("/tmp") / f"manga-ocr-{id(mocr)}-{abs(hash(image_bytes))}.png"
    try:
        tmp.write_bytes(image_bytes)
        return dict(mocr(str(tmp)))
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()

    async def recognize_text(self, image_bytes: bytes) -> str:
        if not self._ready:
            raise RuntimeError(self._error or "ocr engine not ready")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._mocr.mocr, pil_image)


def engine_factory(name: str, *, force_cpu: bool = True) -> OcrEngine:
    if name == "mokuro":
        return MokuroEngine(force_cpu=force_cpu)
    raise ValueError(f"unknown ocr backend: {name}")


def is_allowed_image_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme.lower() in ALLOWED_IMAGE_URL_SCHEMES and bool(parsed.netloc)


async def fetch_image_url(url: str, *, client: httpx.AsyncClient | None = None) -> bytes:
    if not is_allowed_image_url(url):
        raise ValueError("image_url must be an http(s) URL")
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        return bytes(response.content)
    finally:
        if owns_client:
            await http_client.aclose()


def rewrite_image_url(url: str, manga_server_url: str) -> str:
    """Rewrite an image URL's scheme/host to manga_server_url if configured.

    The browser sends URLs pointing at the WebUI host (e.g. ``http://localhost:3000/api/v1/...``).
    When the OCR backend runs in a separate container it cannot resolve those URLs, so we
    replace the scheme/host[:port] with ``manga_server_url`` when set.
    """
    if not manga_server_url:
        return url
    parsed_target = urlparse(manga_server_url)
    if not parsed_target.scheme or not parsed_target.netloc:
        return url
    parsed_source = urlparse(url)
    if parsed_source.scheme.lower() not in ALLOWED_IMAGE_URL_SCHEMES:
        return url
    return parsed_target._replace(path=parsed_source.path, query=parsed_source.query, fragment=parsed_source.fragment).geturl()


def decode_image(
    b64: str | None,
    url: str | None,
    *,
    fetcher: Callable[[str], bytes] | None = None,
) -> bytes:
    if b64:
        decoded = base64.b64decode(b64)
        return bytes(decoded)
    if url:
        if fetcher is None:
            raise ValueError("image_url requires a fetcher")
        return fetcher(url)
    raise ValueError("either image_base64 or image_url required")


async def decode_image_async(
    b64: str | None,
    url: str | None,
    *,
    http_client: httpx.AsyncClient | None = None,
    manga_server_url: str = "",
) -> bytes:
    if b64:
        decoded = base64.b64decode(b64)
        return bytes(decoded)
    if url:
        effective_url = rewrite_image_url(url, manga_server_url) if manga_server_url else url
        return await fetch_image_url(effective_url, client=http_client)
    raise ValueError("either image_base64 or image_url required")
