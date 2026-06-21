from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from manga_learning_service.cache.keys import PageIdentity, hash_bytes


@pytest.fixture
def sample_png_bytes() -> bytes:
    image = Image.new("RGB", (800, 1200), color="white")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_hash_bytes_is_deterministic(sample_png_bytes: bytes) -> None:
    assert hash_bytes(sample_png_bytes) == hash_bytes(sample_png_bytes)


def test_hash_bytes_differs_for_different_input(sample_png_bytes: bytes) -> None:
    other = Image.new("RGB", (100, 100), color="black")
    buf = BytesIO()
    other.save(buf, format="PNG")
    assert hash_bytes(sample_png_bytes) != hash_bytes(buf.getvalue())


def test_page_identity_cache_key_changes_with_language(sample_png_bytes: bytes) -> None:
    base = PageIdentity(
        manga_id="m1",
        chapter_id="c1",
        page_index=0,
        image_url=None,
        image_hash=hash_bytes(sample_png_bytes),
        language="ja",
        backend="mokuro",
    )
    other = PageIdentity(
        manga_id="m1",
        chapter_id="c1",
        page_index=0,
        image_url=None,
        image_hash=hash_bytes(sample_png_bytes),
        language="en",
        backend="mokuro",
    )
    assert base.cache_key() != other.cache_key()


def test_page_identity_cache_key_changes_with_backend(sample_png_bytes: bytes) -> None:
    base = PageIdentity(
        manga_id="m1",
        chapter_id="c1",
        page_index=0,
        image_url=None,
        image_hash=hash_bytes(sample_png_bytes),
        language="ja",
        backend="mokuro",
    )
    other = PageIdentity(
        manga_id="m1",
        chapter_id="c1",
        page_index=0,
        image_url=None,
        image_hash=hash_bytes(sample_png_bytes),
        language="ja",
        backend="lens",
    )
    assert base.cache_key() != other.cache_key()
