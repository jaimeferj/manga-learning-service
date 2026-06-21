from __future__ import annotations

from typing import Protocol


class AiProvider(Protocol):
    name: str

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> str: ...
