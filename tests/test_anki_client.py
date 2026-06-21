from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from manga_learning_service.anki.client import AnkiConnect, AnkiError


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(json.loads(request.read().decode("utf-8")))
        return httpx.Response(200, json=self.response)


@pytest.fixture
def make_client():
    def _factory(response: dict[str, Any]) -> tuple[AnkiConnect, _Transport]:
        transport = _Transport(response)
        http = httpx.AsyncClient(transport=transport, base_url="http://anki.test")
        client = AnkiConnect("http://anki.test", client=http)
        return client, transport

    return _factory


async def test_version(make_client) -> None:
    client, transport = make_client({"result": 6, "error": None})
    assert await client.version() == 6
    assert transport.calls[0]["action"] == "version"


async def test_deck_names(make_client) -> None:
    client, transport = make_client({"result": ["Default", "Mining"], "error": None})
    decks = await client.deck_names()
    assert decks == ["Default", "Mining"]


async def test_add_note_sends_correct_payload(make_client) -> None:
    client, transport = make_client({"result": 42, "error": None})
    note_id = await client.add_note(
        deck="Mining",
        model="Japanese",
        fields={"Expression": "猫", "Reading": "ねこ", "Meaning": "cat"},
        tags=["manga"],
    )
    assert note_id == 42
    call = transport.calls[0]
    assert call["action"] == "addNote"
    assert call["params"]["note"]["deckName"] == "Mining"
    assert call["params"]["note"]["fields"]["Expression"] == "猫"
    assert call["params"]["note"]["tags"] == ["manga"]
    await client.aclose()


async def test_anki_error_propagates(make_client) -> None:
    client, _ = make_client({"result": None, "error": "deck not found"})
    with pytest.raises(AnkiError, match="deck not found"):
        await client.version()
    await client.aclose()
