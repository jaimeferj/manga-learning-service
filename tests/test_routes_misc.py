from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from manga_learning_service.main import create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MANGA_LEARNING_CACHE_DB_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("MANGA_LEARNING_AI_ENABLED", "true")
    monkeypatch.setenv("MANGA_LEARNING_AI_PROVIDER", "openai")
    monkeypatch.setenv("MANGA_LEARNING_AI_OPENAI_API_KEY", "test")
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def test_ai_status(client: TestClient) -> None:
    response = client.get("/ai/status")
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "openai"
    assert data["enabled"] == "true"


def test_anki_status_disabled(client: TestClient) -> None:
    response = client.get("/anki/status")
    assert response.status_code == 200
    assert response.json()["enabled"] is False


def test_anki_create_card_blocked_when_disabled(client: TestClient) -> None:
    response = client.post(
        "/anki/create-card",
        json={"deck": "Mining", "model": "Basic", "fields": {"Front": "x", "Back": "y"}},
    )
    assert response.status_code == 503
