from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import aiosqlite

from manga_learning_service.cache.keys import PageIdentity

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr_pages (
    cache_key TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    language TEXT NOT NULL,
    manga_id TEXT,
    chapter_id TEXT,
    page_index INTEGER,
    image_url TEXT,
    image_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ocr_pages_manga ON ocr_pages(manga_id);
CREATE INDEX IF NOT EXISTS idx_ocr_pages_chapter ON ocr_pages(manga_id, chapter_id);
"""


class CacheStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        if self._db is not None:
            return
        path = Path(self._db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("cache store not initialised")
        return self._db

    async def get_page(self, identity: PageIdentity) -> dict[str, Any] | None:
        cursor = await self.db.execute(
            "SELECT result_json FROM ocr_pages WHERE cache_key = ?",
            (identity.cache_key(),),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            return None
        return cast(dict[str, Any], payload)

    async def put_page(self, identity: PageIdentity, result: dict[str, Any]) -> None:
        await self.db.execute(
            """
            INSERT OR REPLACE INTO ocr_pages
                (cache_key, backend, language, manga_id, chapter_id, page_index,
                 image_url, image_hash, result_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identity.cache_key(),
                identity.backend,
                identity.language,
                identity.manga_id,
                identity.chapter_id,
                identity.page_index,
                identity.image_url,
                identity.image_hash,
                json.dumps(result),
                datetime.now(UTC).isoformat(),
            ),
        )
        await self.db.commit()

    async def purge(
        self,
        manga_id: str | None = None,
        chapter_id: str | None = None,
    ) -> int:
        if manga_id is None and chapter_id is None:
            cursor = await self.db.execute("DELETE FROM ocr_pages")
        elif chapter_id is None:
            cursor = await self.db.execute(
                "DELETE FROM ocr_pages WHERE manga_id = ?",
                (manga_id,),
            )
        else:
            cursor = await self.db.execute(
                "DELETE FROM ocr_pages WHERE manga_id = ? AND chapter_id = ?",
                (manga_id, chapter_id),
            )
        removed = cursor.rowcount
        await self.db.commit()
        await cursor.close()
        return removed

    async def count(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM ocr_pages")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0]) if row else 0
