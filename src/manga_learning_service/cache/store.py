from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
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
    persistent INTEGER NOT NULL DEFAULT 0,
    expires_at TEXT,
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
        await self._migrate()

    async def _migrate(self) -> None:
        cursor = await self.db.execute("PRAGMA table_info(ocr_pages)")
        columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        if "persistent" not in columns:
            await self.db.execute(
                "ALTER TABLE ocr_pages ADD COLUMN persistent INTEGER NOT NULL DEFAULT 0"
            )
        if "expires_at" not in columns:
            await self.db.execute("ALTER TABLE ocr_pages ADD COLUMN expires_at TEXT")
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocr_pages_expires ON ocr_pages(expires_at)"
        )
        await self.db.commit()

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
        await self._cleanup_expired()
        cursor = await self.db.execute(
            "SELECT result_json, persistent, expires_at FROM ocr_pages WHERE cache_key = ?",
            (identity.cache_key(),),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        result_json, _persistent, _expires_at = row
        payload = json.loads(result_json)
        if not isinstance(payload, dict):
            return None
        return cast(dict[str, Any], payload)

    async def put_page(
        self,
        identity: PageIdentity,
        result: dict[str, Any],
        *,
        persistent: bool = False,
        ttl_seconds: int | None = None,
    ) -> None:
        now = datetime.now(UTC)
        expires_at: str | None = None
        if persistent:
            expires_at = None
        elif ttl_seconds is not None:
            # ttl_seconds <= 0 means "already expired": we still cache the row
            # so existing cache-miss semantics are preserved, but it will be
            # purged on the next read because expires_at <= now.
            expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        await self.db.execute(
            """
            INSERT OR REPLACE INTO ocr_pages
                (cache_key, backend, language, manga_id, chapter_id, page_index,
                 image_url, image_hash, result_json, persistent, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                1 if persistent else 0,
                expires_at,
                now.isoformat(),
            ),
        )
        await self.db.commit()

    async def mark_persistent(self, identity: PageIdentity) -> bool:
        """Promote an existing cache row to persistent (no expiry). Returns True when a row was updated."""
        cursor = await self.db.execute(
            "UPDATE ocr_pages SET persistent = 1, expires_at = NULL WHERE cache_key = ?",
            (identity.cache_key(),),
        )
        updated = cursor.rowcount > 0
        await self.db.commit()
        await cursor.close()
        return updated

    async def _cleanup_expired(self) -> int:
        now = datetime.now(UTC).isoformat()
        cursor = await self.db.execute(
            "DELETE FROM ocr_pages WHERE persistent = 0 AND expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        removed = cursor.rowcount
        await self.db.commit()
        await cursor.close()
        return removed

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
        await self._cleanup_expired()
        cursor = await self.db.execute("SELECT COUNT(*) FROM ocr_pages")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0]) if row else 0