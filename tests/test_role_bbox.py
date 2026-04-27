"""Tests for scripts/role_bbox.py — bbox detection + per-panel cropping.

The `_claude.call` subprocess wrapper is mocked. PIL is real but works
on tiny synthetic images so the tests stay fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from _claude import ClaudeResponse
from role_bbox import (
    BBOX_JSON_SCHEMA,
    PanelBbox,
    crop_panel,
    detect_panels,
)


def _make_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (200, 100, 50)).save(path, "JPEG", quality=85)


def _stub(payload: object) -> ClaudeResponse:
    return ClaudeResponse(
        text=json.dumps(payload), cached=False, request_sha="x"
    )


# ---------------------------------------------------------------------------
# detect_panels
# ---------------------------------------------------------------------------


def test_detect_panels_parses_haiku_response(tmp_path: Path) -> None:
    img = tmp_path / "photo.jpg"
    _make_image(img, (3000, 4000))
    fake = _stub(
        {
            "panels": [
                {
                    "kind": "nutrition",
                    "x_min_frac": 0.13,
                    "y_min_frac": 0.01,
                    "x_max_frac": 0.87,
                    "y_max_frac": 0.63,
                    "text_direction": "horizontal",
                },
                {
                    "kind": "ingredients",
                    "x_min_frac": 0.13,
                    "y_min_frac": 0.63,
                    "x_max_frac": 0.87,
                    "y_max_frac": 0.93,
                    "text_direction": "horizontal",
                },
            ]
        }
    )
    with patch("role_bbox.call", return_value=fake):
        with patch("role_bbox.resize_to_longest_side", return_value=img):
            panels = detect_panels(img)
    assert len(panels) == 2
    assert panels[0].kind == "nutrition"
    assert panels[1].kind == "ingredients"
    assert panels[0].text_direction == "horizontal"


def test_detect_panels_uses_haiku_by_default(tmp_path: Path) -> None:
    """We assume haiku unless the caller overrides; cost-sensitive defaults
    matter because this runs once per photo."""
    img = tmp_path / "photo.jpg"
    _make_image(img, (1000, 1000))
    with patch(
        "role_bbox.call", return_value=_stub({"panels": []})
    ) as mock_call:
        with patch("role_bbox.resize_to_longest_side", return_value=img):
            detect_panels(img)
    request = mock_call.call_args.args[0]
    assert request.model == "haiku"


def test_detect_panels_runs_on_resized_input(tmp_path: Path) -> None:
    """Bbox detection runs at 1024 px by default, not full resolution."""
    img = tmp_path / "photo.jpg"
    _make_image(img, (3000, 4000))
    resized = tmp_path / "resized.jpg"
    _make_image(resized, (768, 1024))
    with patch("role_bbox.call", return_value=_stub({"panels": []})):
        with patch(
            "role_bbox.resize_to_longest_side", return_value=resized
        ) as mock_resize:
            detect_panels(img, detect_at_longest_side=1024)
    assert mock_resize.call_args.args[1] == 1024


def test_detect_panels_handles_empty_panels(tmp_path: Path) -> None:
    """A photo with no readable panels (blurry / unrelated subject) returns ()."""
    img = tmp_path / "photo.jpg"
    _make_image(img, (1000, 1000))
    with patch("role_bbox.call", return_value=_stub({"panels": []})):
        with patch("role_bbox.resize_to_longest_side", return_value=img):
            panels = detect_panels(img)
    assert panels == ()


# ---------------------------------------------------------------------------
# PanelBbox.to_pixels
# ---------------------------------------------------------------------------


def test_to_pixels_converts_fractional_coords() -> None:
    panel = PanelBbox(
        kind="nutrition",
        x_min_frac=0.1,
        y_min_frac=0.2,
        x_max_frac=0.9,
        y_max_frac=0.8,
        text_direction="horizontal",
    )
    assert panel.to_pixels(1000, 2000) == (100, 400, 900, 1600)


# ---------------------------------------------------------------------------
# crop_panel
# ---------------------------------------------------------------------------


def test_crop_panel_writes_jpeg_at_expected_size(tmp_path: Path) -> None:
    img = tmp_path / "photo.jpg"
    _make_image(img, (1000, 2000))
    panel = PanelBbox(
        kind="nutrition",
        x_min_frac=0.1,
        y_min_frac=0.2,
        x_max_frac=0.9,
        y_max_frac=0.8,
        text_direction="horizontal",
    )
    out = crop_panel(img, panel, out_dir=tmp_path / "crops")
    assert out.exists()
    with Image.open(out) as cropped:
        assert cropped.size == (800, 1200)


def test_crop_panel_filename_encodes_coords_and_kind(tmp_path: Path) -> None:
    """The filename is deterministic given the source + bbox so re-runs
    are idempotent and downstream callers can predict the path."""
    img = tmp_path / "photo.jpg"
    _make_image(img, (1000, 1000))
    panel = PanelBbox(
        kind="price-tag",
        x_min_frac=0.123,
        y_min_frac=0.234,
        x_max_frac=0.876,
        y_max_frac=0.987,
        text_direction="horizontal",
    )
    out = crop_panel(img, panel, out_dir=tmp_path / "crops")
    # Filename includes the panel kind and the four coords scaled to 0–1000.
    assert "price-tag" in out.name
    assert "0123" in out.name
    assert "0234" in out.name
    assert "0876" in out.name
    assert "0987" in out.name


def test_crop_panel_is_idempotent(tmp_path: Path) -> None:
    """Second call returns the same path without re-encoding."""
    img = tmp_path / "photo.jpg"
    _make_image(img, (1000, 1000))
    panel = PanelBbox(
        kind="front",
        x_min_frac=0.0,
        y_min_frac=0.0,
        x_max_frac=1.0,
        y_max_frac=1.0,
        text_direction="horizontal",
    )
    out1 = crop_panel(img, panel, out_dir=tmp_path / "crops")
    mtime1 = out1.stat().st_mtime_ns
    out2 = crop_panel(img, panel, out_dir=tmp_path / "crops")
    assert out2 == out1
    # The cached path is returned without rewriting the file.
    assert out2.stat().st_mtime_ns == mtime1


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_bbox_schema_pins_role_kinds_and_directions() -> None:
    """The schema enums are how the model knows which strings are valid;
    if someone edits ROLE_KINDS we want this test to flag the schema
    didn't get updated alongside."""
    schema = json.loads(BBOX_JSON_SCHEMA)
    panel_props = schema["properties"]["panels"]["items"]["properties"]
    assert set(panel_props["kind"]["enum"]) == {
        "front",
        "nutrition",
        "ingredients",
        "other-label",
        "price-tag",
    }
    assert set(panel_props["text_direction"]["enum"]) == {
        "horizontal",
        "vertical",
    }
