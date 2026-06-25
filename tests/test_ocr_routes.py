from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from manga_learning_service.config import Settings
from manga_learning_service.main import create_app
from manga_learning_service.ocr.service import rewrite_image_url


class StubEngine:
    name = "stub"
    ready = True
    error = None

    def __init__(self) -> None:
        self.calls = 0

    async def recognize_page(self, image_bytes: bytes) -> dict[str, Any]:
        self.calls += 1
        return {
            "img_width": 800,
            "img_height": 1200,
            "blocks": [
                {
                    "box": [0, 0, 400, 200],
                    "vertical": True,
                    "font_size": 24,
                    "lines": ["こんにちは"],
                    "lines_coords": [[[0, 0], [400, 0], [400, 200], [0, 200]]],
                },
                {
                    "box": [0, 200, 400, 400],
                    "vertical": False,
                    "font_size": 18,
                    "lines": ["world"],
                    "lines_coords": [[[0, 0], [400, 0], [400, 200], [0, 200]]],
                },
            ],
        }

    async def recognize_text(self, image_bytes: bytes) -> str:
        return "こんにちは"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MANGA_LEARNING_CACHE_DB_PATH", str(tmp_path / "cache.db"))
    application = create_app()
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def _patch_engine(monkeypatch, app) -> StubEngine:
    from manga_learning_service.ocr import routes

    stub = StubEngine()
    monkeypatch.setattr(routes, "engine_factory", lambda *_a, **_kw: stub)
    monkeypatch.setattr(routes, "_engine", stub)
    return stub


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ocr_status(client: TestClient, monkeypatch) -> None:
    _patch_engine(monkeypatch, client.app)
    response = client.get("/ocr/status")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["backend"] == "mokuro"


def test_ocr_page_returns_normalized_lines(client: TestClient, monkeypatch) -> None:
    _patch_engine(monkeypatch, client.app)

    image = Image.new("RGB", (800, 1200), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    payload = {
        "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "language": "ja",
        "manga_id": "m1",
        "chapter_id": "c1",
        "page_index": 0,
    }
    response = client.post("/ocr/page", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["img_width"] == 800
    assert data["img_height"] == 1200
    assert len(data["lines"]) == 2
    first = data["lines"][0]
    assert first["text"] == "こんにちは"
    assert first["forcedOrientation"] == "vertical"
    assert first["tightBoundingBox"]["x"] == 0.0
    assert first["tightBoundingBox"]["y"] == 0.0
    assert first["tightBoundingBox"]["width"] == pytest.approx(0.5)
    assert first["tightBoundingBox"]["height"] == pytest.approx(200 / 1200)
    assert data["cached"] is False


def test_ocr_page_uses_cache_on_second_call(client: TestClient, monkeypatch) -> None:
    stub = _patch_engine(monkeypatch, client.app)

    image = Image.new("RGB", (800, 1200), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    payload = {
        "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "manga_id": "m1",
        "chapter_id": "c1",
        "page_index": 0,
    }
    first = client.post("/ocr/page", json=payload).json()
    second = client.post("/ocr/page", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert stub.calls == 1


def test_ocr_page_fetches_image_url(client: TestClient, monkeypatch) -> None:
    stub = _patch_engine(monkeypatch, client.app)

    image = Image.new("RGB", (800, 1200), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=image_bytes, headers={"content-type": "image/png"})

    transport = httpx.MockTransport(handler)
    from manga_learning_service.ocr import routes

    async def fake_get_http_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, timeout=10.0, follow_redirects=True)

    monkeypatch.setattr(routes, "get_http_client", fake_get_http_client)

    response = client.post(
        "/ocr/page",
        json={
            "image_url": "https://example.com/page-1.png",
            "manga_id": "m1",
            "chapter_id": "c1",
            "page_index": 0,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["cached"] is False
    assert len(data["lines"]) == 2
    assert stub.calls == 1


def test_ocr_page_rejects_non_http_url(client: TestClient, monkeypatch) -> None:
    _patch_engine(monkeypatch, client.app)
    response = client.post("/ocr/page", json={"image_url": "file:///etc/passwd"})
    assert response.status_code == 400
    assert "http" in response.json()["detail"].lower()


def test_ocr_page_requires_image_input(client: TestClient, monkeypatch) -> None:
    _patch_engine(monkeypatch, client.app)
    response = client.post("/ocr/page", json={})
    assert response.status_code == 400


def test_ocr_recognize_region_returns_text(client: TestClient, monkeypatch) -> None:
    _patch_engine(monkeypatch, client.app)
    image = Image.new("RGB", (400, 400), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    payload = {
        "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "region": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
    }
    response = client.post("/ocr/recognize-region", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["text"] == "こんにちは"
    assert data["tightBoundingBox"] == payload["region"]


def test_ocr_recognize_region_fetches_image_url(client: TestClient, monkeypatch) -> None:
    _patch_engine(monkeypatch, client.app)

    image = Image.new("RGB", (400, 400), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=image_bytes, headers={"content-type": "image/png"})

    transport = httpx.MockTransport(handler)
    from manga_learning_service.ocr import routes

    async def fake_get_http_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, timeout=10.0, follow_redirects=True)

    monkeypatch.setattr(routes, "get_http_client", fake_get_http_client)

    response = client.post(
        "/ocr/recognize-region",
        json={
            "image_url": "https://example.com/region.png",
            "region": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "こんにちは"


def test_ocr_purge_cache(client: TestClient, monkeypatch) -> None:
    _patch_engine(monkeypatch, client.app)
    image = Image.new("RGB", (400, 400), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    client.post("/ocr/page", json={"image_base64": encoded, "manga_id": "m1", "page_index": 0})
    client.post("/ocr/page", json={"image_base64": encoded, "manga_id": "m2", "page_index": 0})
    response = client.post("/ocr/purge-cache", json={"manga_id": "m1"})
    assert response.status_code == 200
    assert response.json()["removed"] >= 1


def test_rewrite_image_url_replaces_host() -> None:
    assert (
        rewrite_image_url("http://localhost:3000/api/v1/manga/1/page/0?sourceId=x", "http://suwayomi-server:4567")
        == "http://suwayomi-server:4567/api/v1/manga/1/page/0?sourceId=x"
    )


def test_rewrite_image_url_no_op_when_empty() -> None:
    url = "http://localhost:3000/api/v1/manga/1/page/0"
    assert rewrite_image_url(url, "") == url


def test_ocr_page_rewrites_image_url_when_manga_server_url_set(monkeypatch, tmp_path) -> None:
    image = Image.new("RGB", (400, 400), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    seen_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, content=image_bytes, headers={"content-type": "image/png"})

    transport = httpx.MockTransport(handler)

    monkeypatch.setenv("MANGA_LEARNING_CACHE_DB_PATH", str(tmp_path / "cache.db"))

    from manga_learning_service.ocr import routes as ocr_routes

    monkeypatch.setattr(
        "manga_learning_service.ocr.routes.get_settings",
        lambda: Settings(manga_server_url="http://suwayomi-server:4567"),
    )

    async def fake_get_http_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, timeout=10.0, follow_redirects=True)

    monkeypatch.setattr(ocr_routes, "get_http_client", fake_get_http_client)

    stub = StubEngine()
    monkeypatch.setattr(ocr_routes, "engine_factory", lambda *_a, **_kw: stub)
    monkeypatch.setattr(ocr_routes, "_engine", stub)

    app = create_app()
    with TestClient(app) as c:
        response = c.post(
            "/ocr/page",
            json={
                "image_url": "http://localhost:3000/api/v1/manga/1/page/0?sourceId=x",
                "manga_id": "m1",
                "page_index": 0,
            },
        )
    assert response.status_code == 200, response.text
    assert seen_urls == ["http://suwayomi-server:4567/api/v1/manga/1/page/0?sourceId=x"]


def test_settings_has_manga_server_url_default() -> None:
    settings = Settings()
    assert hasattr(settings, "manga_server_url")
    assert settings.manga_server_url == ""


def test_settings_exposes_ocr_page_ratio_range() -> None:
    settings = Settings()
    assert settings.ocr_page_ratio_min == 1.25
    assert settings.ocr_page_ratio_max == 1.85
    assert settings.ocr_min_pages == 8
    assert settings.ocr_max_pages == 25
    assert 0.0 < settings.ocr_virtual_page_overlap < 1.0
    assert 0.0 < settings.ocr_iou_dedup_threshold < 1.0
