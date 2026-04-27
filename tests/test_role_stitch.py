"""Tests for scripts/role_stitch.py — per-role stitching with text-axis."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from role_bbox import PanelBbox
from role_stitch import (
    SEPARATOR_COLOR,
    SEPARATOR_PX,
    stitch_axis_for_panels,
    stitch_role_panels,
)


def _save(
    path: Path, size: tuple[int, int], color: tuple[int, int, int]
) -> None:
    Image.new("RGB", size, color).save(path, "JPEG", quality=85)


def _panel(text_direction: str) -> PanelBbox:
    return PanelBbox(
        kind="nutrition",
        x_min_frac=0.0,
        y_min_frac=0.0,
        x_max_frac=1.0,
        y_max_frac=1.0,
        text_direction=text_direction,
    )


# ---------------------------------------------------------------------------
# stitch_axis_for_panels
# ---------------------------------------------------------------------------


def test_stitch_axis_majority_wins() -> None:
    assert (
        stitch_axis_for_panels(
            [_panel("vertical"), _panel("vertical"), _panel("horizontal")]
        )
        == "vertical"
    )


def test_stitch_axis_tie_breaks_horizontal() -> None:
    """Most product packaging is photographed upright with horizontal text;
    on a tie we should default that way rather than guess."""
    assert (
        stitch_axis_for_panels([_panel("horizontal"), _panel("vertical")])
        == "horizontal"
    )


def test_stitch_axis_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one panel"):
        stitch_axis_for_panels([])


# ---------------------------------------------------------------------------
# stitch_role_panels — horizontal
# ---------------------------------------------------------------------------


def test_stitch_horizontal_concat_widths_with_separator(tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _save(a, (100, 200), (255, 0, 0))
    _save(b, (150, 200), (0, 0, 255))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([a, b], "horizontal", out)
    with Image.open(out) as result:
        assert result.size == (100 + SEPARATOR_PX + 150, 200)


def test_stitch_horizontal_centers_shorter_panel_vertically(
    tmp_path: Path,
) -> None:
    """A short panel next to a tall one should be vertically centered on
    the separator-color canvas — that's the visual anchor that tells the
    extractor "these two strips are the same logical row of text"."""
    short = tmp_path / "short.jpg"
    tall = tmp_path / "tall.jpg"
    _save(short, (100, 100), (255, 0, 0))
    _save(tall, (100, 300), (0, 255, 0))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([short, tall], "horizontal", out)
    with Image.open(out) as result:
        # The top-left of the short panel should be 100px down on the canvas
        # (canvas is 300 tall, panel is 100 tall, so y=100 leaves 100px above
        # and 100px below). Sample the pixel at (50, 50) — should be background.
        bg_pixel = result.getpixel((50, 50))
        red_pixel = result.getpixel((50, 150))
        assert bg_pixel == SEPARATOR_COLOR
        assert red_pixel == (255, 0, 0)


# ---------------------------------------------------------------------------
# stitch_role_panels — vertical
# ---------------------------------------------------------------------------


def test_stitch_vertical_concat_heights_with_separator(tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _save(a, (200, 100), (255, 0, 0))
    _save(b, (200, 150), (0, 0, 255))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([a, b], "vertical", out)
    with Image.open(out) as result:
        assert result.size == (200, 100 + SEPARATOR_PX + 150)


def test_stitch_vertical_centers_narrower_panel_horizontally(
    tmp_path: Path,
) -> None:
    narrow = tmp_path / "narrow.jpg"
    wide = tmp_path / "wide.jpg"
    _save(narrow, (100, 100), (255, 0, 0))
    _save(wide, (300, 100), (0, 255, 0))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([narrow, wide], "vertical", out)
    with Image.open(out) as result:
        # Narrow panel should be horizontally centered: canvas width 300,
        # panel width 100, so x=100 leaves 100 background on each side.
        assert result.getpixel((50, 50)) == SEPARATOR_COLOR
        assert result.getpixel((150, 50)) == (255, 0, 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_stitch_single_panel_writes_unchanged_size(tmp_path: Path) -> None:
    """One panel = no separator, output size matches input size."""
    a = tmp_path / "a.jpg"
    _save(a, (123, 234), (255, 0, 0))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([a], "horizontal", out)
    with Image.open(out) as result:
        assert result.size == (123, 234)


def test_stitch_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one panel"):
        stitch_role_panels([], "horizontal", tmp_path / "out.jpg")


def test_stitch_rejects_unknown_direction(tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"
    _save(a, (100, 100), (255, 0, 0))
    with pytest.raises(ValueError, match="must be 'horizontal' or 'vertical'"):
        stitch_role_panels([a], "diagonal", tmp_path / "out.jpg")


def test_stitch_creates_output_directory_if_missing(tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"
    _save(a, (100, 100), (255, 0, 0))
    out = tmp_path / "nested" / "dir" / "stitched.jpg"
    stitch_role_panels([a], "horizontal", out)
    assert out.exists()
