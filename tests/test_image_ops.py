"""Tests for scripts/_image_ops.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from _image_ops import longest_side_px, resize_to_longest_side
from PIL import Image


def _solid_image(path: Path, size: tuple[int, int]) -> None:
    """Write a solid-color JPEG of the given size to ``path``."""
    Image.new("RGB", size, (200, 50, 50)).save(path, "JPEG")


def test_resize_shrinks_largest_side(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (4000, 2000))
    cache = tmp_path / "cache"

    out = resize_to_longest_side(src, 1024, cache)

    assert out.exists()
    assert longest_side_px(out) == 1024


def test_resize_does_not_upscale(tmp_path: Path) -> None:
    """Pillow.thumbnail never upscales — we rely on that."""
    src = tmp_path / "src.jpg"
    _solid_image(src, (200, 100))
    cache = tmp_path / "cache"

    out = resize_to_longest_side(src, 4096, cache)

    assert longest_side_px(out) == 200


def test_resize_is_idempotent_on_rerun(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (1000, 500))
    cache = tmp_path / "cache"

    first = resize_to_longest_side(src, 256, cache)
    first_mtime = first.stat().st_mtime
    second = resize_to_longest_side(src, 256, cache)

    assert first == second
    assert second.stat().st_mtime == first_mtime  # not re-encoded


def test_resize_creates_cache_dir(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (1000, 500))
    cache = tmp_path / "deep" / "cache"
    assert not cache.exists()

    resize_to_longest_side(src, 256, cache)

    assert cache.exists()


def test_resize_rejects_non_positive_size(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (200, 200))
    cache = tmp_path / "cache"

    with pytest.raises(ValueError, match="longest_side"):
        resize_to_longest_side(src, 0, cache)


def test_longest_side_px(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (640, 480))
    assert longest_side_px(src) == 640
