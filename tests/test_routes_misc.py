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


def test_ai_learn_builds_focused_prompt(client: TestClient, monkeypatch) -> None:
    from manga_learning_service.ai import routes

    class StubProvider:
        name = "stub"
        system = ""
        user = ""

        async def complete(self, *, system, user, **_kwargs):
            self.system = system
            self.user = user
            return '{"sections":[{"label":"Traducción natural","content":"Tengo que ir."}]}'

    provider = StubProvider()
    monkeypatch.setattr(routes, "build_provider", lambda _settings: provider)
    response = client.post(
        "/ai/learn",
        json={
            "sentence": "行かなきゃ。",
            "action": "translate",
            "context": {"previous_line": "もう遅いよ。", "speaker": "ユキ"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "provider": "stub",
        "action": "translate",
        "sentence": "行かなきゃ。",
        "sections": [{"label": "Traducción natural", "content": "Tengo que ir."}],
    }
    assert "Spanish-speaking" in provider.system
    assert "Previous line: 「もう遅いよ。」" in provider.user
    assert "Speaker: ユキ" in provider.user
    assert "Traducción literal" in provider.user


def test_ai_learn_rejects_unknown_action(client: TestClient) -> None:
    response = client.post("/ai/learn", json={"sentence": "行く。", "action": "everything"})
    assert response.status_code == 422


def test_ai_learn_rejects_empty_sections(client: TestClient, monkeypatch) -> None:
    from manga_learning_service.ai import routes

    class StubProvider:
        name = "stub"

        async def complete(self, **_kwargs):
            return '{"sections":[]}'

    monkeypatch.setattr(routes, "build_provider", lambda _settings: StubProvider())
    response = client.post("/ai/learn", json={"sentence": "行く。", "action": "grammar"})
    assert response.status_code == 502


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
