from __future__ import annotations

from typing import Any

import httpx


class AnkiError(RuntimeError):
    pass


class AnkiConnect:
    def __init__(
        self,
        url: str,
        *,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._timeout = timeout
        self._client = client

    async def version(self) -> int:
        return int(await self._invoke("version", {}))

    async def deck_names(self) -> list[str]:
        return list(await self._invoke("deckNames", {}))

    async def model_names(self) -> list[str]:
        return list(await self._invoke("modelNames", {}))

    async def model_field_names(self, model: str) -> list[str]:
        return list(await self._invoke("modelFieldNames", {"modelName": model}))

    async def add_note(
        self,
        *,
        deck: str,
        model: str,
        fields: dict[str, str],
        tags: list[str] | None = None,
        picture: dict[str, Any] | None = None,
    ) -> int:
        note: dict[str, Any] = {
            "deckName": deck,
            "modelName": model,
            "fields": fields,
        }
        if tags:
            note["tags"] = tags
        if picture:
            note["picture"] = picture
        return int(await self._invoke("addNote", {"note": note}))

    async def _invoke(self, action: str, params: dict[str, Any]) -> Any:
        payload = {"action": action, "version": 6, "params": params}
        if self._client is not None:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
            data = response.json()
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload)
                response.raise_for_status()
                data = response.json()
        if data.get("error"):
            raise AnkiError(str(data["error"]))
        return data.get("result")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
