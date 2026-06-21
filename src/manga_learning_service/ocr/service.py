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

from PIL import Image

logger = logging.getLogger(__name__)


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
        tmp = Path("/tmp") / f"manga-ocr-{id(self)}.png"
        try:
            tmp.write_bytes(image_bytes)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._mocr, str(tmp))
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
