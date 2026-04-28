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
    extractor "these two strips are the same logical row of text".

    Disable scale matching for this test so the panels keep their
    raw heights (100 vs 300) and the centering math is unambiguous;
    centering is the behavior under test, not the scale match."""
    short = tmp_path / "short.jpg"
    tall = tmp_path / "tall.jpg"
    _save(short, (100, 100), (255, 0, 0))
    _save(tall, (100, 300), (0, 255, 0))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([short, tall], "horizontal", out, match_scale=False)
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
    """Centering on the cross-axis for vertical stitching. Like the
    horizontal version, disable scale matching to keep raw widths
    so the centering offset is unambiguous."""
    narrow = tmp_path / "narrow.jpg"
    wide = tmp_path / "wide.jpg"
    _save(narrow, (100, 100), (255, 0, 0))
    _save(wide, (300, 100), (0, 255, 0))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([narrow, wide], "vertical", out, match_scale=False)
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


# ---------------------------------------------------------------------------
# match_scale — multi-shot scale normalization
# ---------------------------------------------------------------------------


def test_match_scale_downsamples_larger_panels_to_smallest(
    tmp_path: Path,
) -> None:
    """When two panels have very different natural sizes (the multi-shot
    scale-mismatch problem the user flagged: one frame at 3000px, another
    at 800px), the stitcher downsamples the larger one to match. After
    matching, both panels use the same number of pixels per text glyph
    and opus can read them as one coherent panel."""
    big = tmp_path / "big.jpg"
    small = tmp_path / "small.jpg"
    _save(big, (3000, 1500), (255, 0, 0))
    _save(small, (800, 400), (0, 0, 255))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([big, small], "horizontal", out, match_scale=True)
    with Image.open(out) as result:
        # Big downsampled to longest=800 (preserving aspect 2:1) → 800x400.
        # Small unchanged at 800x400. Stitch width = 800 + sep + 800.
        assert result.size == (800 + SEPARATOR_PX + 800, 400)


def test_match_scale_default_is_on(tmp_path: Path) -> None:
    """Default behavior matches scale — match_scale=True is what the
    orchestrator wants and is the safe default for any caller passing
    multiple panels of varying natural sizes."""
    big = tmp_path / "big.jpg"
    small = tmp_path / "small.jpg"
    _save(big, (2000, 1000), (255, 0, 0))
    _save(small, (1000, 500), (0, 0, 255))
    out_default = tmp_path / "default.jpg"
    out_off = tmp_path / "off.jpg"
    stitch_role_panels([big, small], "horizontal", out_default)
    stitch_role_panels([big, small], "horizontal", out_off, match_scale=False)
    with Image.open(out_default) as a, Image.open(out_off) as b:
        # match_scale=True downsamples big to 1000-wide; off keeps 2000.
        assert a.size[0] < b.size[0]


def test_match_scale_skips_when_only_one_panel(tmp_path: Path) -> None:
    """A one-panel call has nothing to match against; match_scale is a
    no-op and the output is the panel itself at its natural size."""
    a = tmp_path / "a.jpg"
    _save(a, (1234, 567), (255, 0, 0))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([a], "horizontal", out, match_scale=True)
    with Image.open(out) as result:
        assert result.size == (1234, 567)


def test_match_scale_off_preserves_raw_sizes(tmp_path: Path) -> None:
    """Disabling match_scale keeps the raw crop sizes — useful for
    debugging or when sizes are already coordinated upstream."""
    big = tmp_path / "big.jpg"
    small = tmp_path / "small.jpg"
    _save(big, (2000, 1000), (255, 0, 0))
    _save(small, (1000, 500), (0, 0, 255))
    out = tmp_path / "stitched.jpg"
    stitch_role_panels([big, small], "horizontal", out, match_scale=False)
    with Image.open(out) as result:
        # Raw sizes: max height = 1000, width = 2000 + sep + 1000.
        assert result.size == (2000 + SEPARATOR_PX + 1000, 1000)
