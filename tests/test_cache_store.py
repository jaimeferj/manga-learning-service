from __future__ import annotations

import pytest

from manga_learning_service.cache.keys import PageIdentity, hash_bytes
from manga_learning_service.cache.store import CacheStore


@pytest.fixture
async def cache(tmp_path):
    store = CacheStore(str(tmp_path / "cache.db"))
    await store.init()
    try:
        yield store
    finally:
        await store.close()


async def test_miss_then_hit(cache: CacheStore, sample_png_bytes: bytes) -> None:
    identity = PageIdentity(
        manga_id="m1",
        chapter_id="c1",
        page_index=0,
        image_url=None,
        image_hash=hash_bytes(sample_png_bytes),
        language="ja",
        backend="mokuro",
    )
    assert await cache.get_page(identity) is None
    payload = {"img_width": 800, "img_height": 1200, "lines": []}
    await cache.put_page(identity, payload)
    cached = await cache.get_page(identity)
    assert cached == payload


async def test_purge_by_manga(cache: CacheStore, sample_png_bytes: bytes) -> None:
    identities = [
        PageIdentity(
            manga_id="m1",
            chapter_id="c1",
            page_index=i,
            image_url=None,
            image_hash=hash_bytes(sample_png_bytes),
            language="ja",
            backend="mokuro",
        )
        for i in range(3)
    ]
    other = PageIdentity(
        manga_id="m2",
        chapter_id="c1",
        page_index=0,
        image_url=None,
        image_hash=hash_bytes(sample_png_bytes),
        language="ja",
        backend="mokuro",
    )
    for ident in identities + [other]:
        await cache.put_page(ident, {"img_width": 1, "img_height": 1, "lines": []})

    removed = await cache.purge(manga_id="m1")
    assert removed == 3
    assert await cache.count() == 1


async def test_purge_by_chapter(cache: CacheStore, sample_png_bytes: bytes) -> None:
    for chap in ("c1", "c2"):
        ident = PageIdentity(
            manga_id="m1",
            chapter_id=chap,
            page_index=0,
            image_url=None,
            image_hash=hash_bytes(sample_png_bytes),
            language="ja",
            backend="mokuro",
        )
        await cache.put_page(ident, {"img_width": 1, "img_height": 1, "lines": []})
    removed = await cache.purge(manga_id="m1", chapter_id="c2")
    assert removed == 1
