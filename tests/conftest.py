from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from manga_learning_service.ocr import routes as ocr_routes


@pytest.fixture(autouse=True)
def reset_ocr_engine() -> None:
    ocr_routes._engine = None


@pytest.fixture
def sample_png_bytes() -> bytes:
    image = Image.new("RGB", (800, 1200), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
