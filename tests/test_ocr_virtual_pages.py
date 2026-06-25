"""Tests for the virtual-page OCR splitter.

The splitter estimates the number of manga pages in a stitched long-strip image
from the image's aspect ratio and a normal manga page ratio range, then slices
the image into that many overlapping virtual pages. These tests pin down the
estimation behaviour and verify the slice/dedupe logic on synthetic data.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from manga_learning_service.ocr.service import (
    DEFAULT_IOU_DEDUP_THRESHOLD,
    DEFAULT_MANGA_PAGE_RATIO_MAX,
    DEFAULT_MANGA_PAGE_RATIO_MIN,
    DEFAULT_MAX_PAGES,
    DEFAULT_MIN_PAGES,
    DEFAULT_VIRTUAL_PAGE_OVERLAP,
    _dedupe_blocks,
    _iou,
    _slice_virtual_pages,
    estimate_page_count,
)

MANGA_PAGE_RATIO_MIN = DEFAULT_MANGA_PAGE_RATIO_MIN
MANGA_PAGE_RATIO_MAX = DEFAULT_MANGA_PAGE_RATIO_MAX
MIN_PAGES = DEFAULT_MIN_PAGES
MAX_PAGES = DEFAULT_MAX_PAGES
VIRTUAL_PAGE_OVERLAP = DEFAULT_VIRTUAL_PAGE_OVERLAP
IOU_DEDUP_THRESHOLD = DEFAULT_IOU_DEDUP_THRESHOLD


def test_estimate_page_count_returns_none_for_normal_aspect() -> None:
    # A 800x1200 page is 1.5x taller than wide - normal single manga page.
    assert estimate_page_count(800, 1200) is None


def test_estimate_page_count_returns_none_for_short_strip() -> None:
    # Anything < ~2.5x width is treated as a single image.
    assert estimate_page_count(800, 1900) is None


def test_estimate_page_count_for_one_piece_chapter_1182() -> None:
    # The actual chapter the user reported: 1403x34816.
    # With the default ratio range (1.25-1.85), both 16 (ratio 1.55) and 17
    # (ratio 1.46) are valid; we accept either.
    n = estimate_page_count(1403, 34816)
    assert n in (16, 17)
    # The inferred virtual page ratio should land in the normal manga range.
    page_ratio = (34816 / n) / 1403
    assert MANGA_PAGE_RATIO_MIN <= page_ratio <= MANGA_PAGE_RATIO_MAX


def test_estimate_page_count_respects_tight_source_ratio() -> None:
    # With One Piece's known 1.55-1.58 ratio, only n=16 (ratio 1.551) fits.
    n = estimate_page_count(1403, 34816, ratio_min=1.55, ratio_max=1.58)
    assert n == 16


def test_estimate_page_count_returns_none_when_no_candidate_fits_ratio() -> None:
    # An impossibly narrow ratio range yields no fit.
    n = estimate_page_count(1403, 34816, ratio_min=1.60, ratio_max=1.61)
    assert n is None


def test_estimate_page_count_respects_min_max_pages() -> None:
    # Restricting the range to 12-20 still picks 16 for this chapter.
    n = estimate_page_count(1403, 34816, min_pages=12, max_pages=20, ratio_min=1.55, ratio_max=1.58)
    assert n == 16


def test_estimate_page_count_handles_invalid_config() -> None:
    # ratio_min >= ratio_max or min_pages > max_pages should not crash; return None.
    assert estimate_page_count(1403, 34816, ratio_min=2.0, ratio_max=1.0) is None
    assert estimate_page_count(1403, 34816, min_pages=20, max_pages=10) is None


def test_estimate_page_count_returns_sane_value_for_a_range_of_widths() -> None:
    # Different source widths should still produce a sensible page count.
    for width in (700, 1000, 1403, 1800, 2400):
        # Construct a height that matches 17 pages of ratio 1.5
        height = int(round(width * 1.5 * 17))
        n = estimate_page_count(width, height)
        assert n is not None
        assert MIN_PAGES <= n <= MAX_PAGES


def test_estimate_page_count_returns_none_for_extreme_aspect() -> None:
    # A 100x100000 image: 17 pages would have aspect 58x; well outside the
    # manga range, so no candidate page count fits and we fall back to None.
    assert estimate_page_count(100, 100_000) is None


def test_slice_virtual_pages_covers_full_height() -> None:
    width = 1403
    page_count = 17
    height = 34816
    arr = np.full((height, width, 3), 32, dtype=np.uint8)
    image = Image.fromarray(arr)
    slices = _slice_virtual_pages(image, page_count)
    assert len(slices) == page_count
    # First slice starts at 0
    assert slices[0][1] == 0
    # Last slice reaches the bottom
    from PIL import Image as _Img

    last_h = _Img.open(BytesIO(slices[-1][0])).height
    last_y = slices[-1][1] + last_h
    assert last_y == height
    # Each slice is strictly within the image bounds
    for slice_bytes, y_offset in slices:
        h = _Img.open(BytesIO(slice_bytes)).height
        assert y_offset >= 0
        assert y_offset + h <= height


def test_slice_virtual_pages_overlaps() -> None:
    # With overlap, adjacent slices must share a vertical band.
    width, height = 1403, 34816
    image = Image.fromarray(np.full((height, width, 3), 0, dtype=np.uint8))
    slices = _slice_virtual_pages(image, 17)
    base_step = height / 17
    for i in range(len(slices) - 1):
        _, y0 = slices[i]
        from PIL import Image as _Img

        h0 = _Img.open(BytesIO(slices[i][0])).height
        y1 = slices[i + 1][1]
        # The next slice starts before the previous one ends by at least 1 pixel
        assert y1 < y0 + h0
        # And the overlap is at least a small fraction of the base step
        assert y0 + h0 - y1 >= int(base_step * 0.05)


def test_iou_zero_for_disjoint_boxes() -> None:
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_one_for_identical_boxes() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_dedupe_blocks_drops_high_iou_duplicates() -> None:
    blocks = [
        {"box": [10, 10, 100, 50], "text": "hello"},
        {"box": [12, 12, 102, 52], "text": "hello"},
        {"box": [500, 500, 600, 540], "text": "world"},
    ]
    kept = _dedupe_blocks(blocks)
    assert len(kept) == 2
    assert kept[0]["text"] == "hello"
    assert kept[1]["text"] == "world"


def test_dedupe_blocks_keeps_distinct_boxes() -> None:
    blocks = [
        {"box": [0, 0, 50, 30], "text": "a"},
        {"box": [0, 40, 50, 70], "text": "b"},
        {"box": [0, 80, 50, 110], "text": "c"},
    ]
    kept = _dedupe_blocks(blocks)
    assert len(kept) == 3


def test_dedupe_blocks_keeps_blocks_without_box() -> None:
    blocks = [{"text": "no-box"}, {"box": [0, 0, 50, 30], "text": "a"}]
    kept = _dedupe_blocks(blocks)
    assert len(kept) == 2
    assert any(b.get("text") == "no-box" for b in kept)


def test_overlap_threshold_is_conservative() -> None:
    # 0.5 is a standard "loosely overlapping" IoU threshold; the test ensures
    # we do not silently weaken it because that would risk keeping duplicates.
    assert IOU_DEDUP_THRESHOLD >= 0.3


def test_overlap_fraction_is_reasonable() -> None:
    # 10% overlap is large enough to capture most text straddling a virtual
    # page boundary, without doubling the OCR cost.
    assert 0.05 <= VIRTUAL_PAGE_OVERLAP <= 0.25


def test_page_count_bounds_are_sensible() -> None:
    # 8-25 covers typical manga chapter lengths.
    assert 4 <= MIN_PAGES <= 12
    assert MAX_PAGES >= 20


def test_manga_page_ratio_range_covers_real_manga() -> None:
    # Standard manga page proportions sit between ~1.25 and ~1.85.
    assert 1.0 < MANGA_PAGE_RATIO_MIN < 1.4
    assert 1.6 < MANGA_PAGE_RATIO_MAX < 2.2
