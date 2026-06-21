from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PageIdentity:
    manga_id: str | None
    chapter_id: str | None
    page_index: int | None
    image_url: str | None
    image_hash: str
    language: str
    backend: str

    def cache_key(self) -> str:
        parts: Iterable[str] = (
            self.backend,
            self.language,
            self.manga_id or "",
            self.chapter_id or "",
            str(self.page_index) if self.page_index is not None else "",
            self.image_url or "",
            self.image_hash,
        )
        joined = "|".join(parts)
        return "page:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
