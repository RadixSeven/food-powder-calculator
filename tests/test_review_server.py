"""Tests for scripts/review_server.py — the multi-view entry point.

Covers the landing page, view-enable rules, redirects, and the CLI
``main`` glue. View-specific behaviour lives in the per-view test
files (test_review_groups, test_review_crops).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from review_server import create_app, main


def _solid_image(path: Path) -> None:
    Image.new("RGB", (40, 40), (200, 50, 50)).save(path, "JPEG")


@pytest.fixture
def fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Build minimal-but-valid inputs for both views.

    Returns ``(groups_json, photos_dir, gold_groups_json, stitched_root,
    crop_reviews_json)``. Tests can pass these straight to ``create_app``
    or stash a subset on disk and feed ``None`` for the rest.
    """
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _solid_image(photos_dir / "a.jpg")
    groups_json = tmp_path / "groups.json"
    groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "g1",
                        "store": "MOM",
                        "photos": [
                            {"path": "photos/a.jpg", "roles": ["front"]}
                        ],
                        "warnings": [],
                        "locked": False,
                    }
                ]
            }
        )
    )
    gold_groups_json = tmp_path / "gold.json"
    gold_groups_json.write_text(
        json.dumps({"groups": [{"id": "g1", "store": "MOM"}]})
    )
    stitched_root = tmp_path / "stitched"
    stitched_root.mkdir()
    crop_reviews_json = tmp_path / "crop_reviews.json"
    return (
        groups_json,
        photos_dir,
        gold_groups_json,
        stitched_root,
        crop_reviews_json,
    )


def test_landing_lists_both_views_when_enabled(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """With both views enabled, the landing page links to each one."""
    groups_json, photos_dir, gold, stitched, crops = fixture_paths
    app = create_app(
        groups_json=groups_json,
        photos_dir=photos_dir,
        gold_groups_json=gold,
        stitched_root=stitched,
        crop_reviews_json=crops,
    )
    body = app.test_client().get("/").data.decode()
    assert 'href="/groups/"' in body
    assert 'href="/crops/"' in body
    # No "disabled" markup when both inputs are present.
    assert "disabled:" not in body


def test_landing_marks_crops_disabled_when_inputs_missing(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """When the crops-view inputs aren't supplied (e.g. a fresh repo
    with no gold-groups), the landing page shows the crops link as
    disabled with a why-message rather than 404-ing on click.
    """
    groups_json, photos_dir, *_ = fixture_paths
    app = create_app(
        groups_json=groups_json,
        photos_dir=photos_dir,
        gold_groups_json=None,
        stitched_root=None,
    )
    body = app.test_client().get("/").data.decode()
    assert 'href="/groups/"' in body
    assert 'href="/crops/"' not in body
    assert "disabled:" in body


def test_root_path_redirects_for_each_view(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """A user typing /groups (no slash) should land on /groups/."""
    groups_json, photos_dir, gold, stitched, crops = fixture_paths
    app = create_app(
        groups_json=groups_json,
        photos_dir=photos_dir,
        gold_groups_json=gold,
        stitched_root=stitched,
        crop_reviews_json=crops,
    )
    client = app.test_client()
    for prefix in ("/groups", "/crops"):
        response = client.get(prefix)
        assert response.status_code == 302
        assert response.headers["Location"].endswith(f"{prefix}/")


def test_create_app_skips_crops_when_either_input_is_none(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """Both crops-view inputs (gold + stitched root) must be set;
    setting only one leaves the view unregistered. Lets a fresh-repo
    smoke test of the groups view succeed without forging crops state.
    """
    groups_json, photos_dir, gold, _stitched, _crops = fixture_paths
    app = create_app(
        groups_json=groups_json,
        photos_dir=photos_dir,
        gold_groups_json=gold,
        stitched_root=None,
    )
    # /crops/ is unregistered → 404 from Flask's URL map.
    assert app.test_client().get("/crops/").status_code == 404
    # /groups/ still works.
    assert app.test_client().get("/groups/").status_code == 200


def test_main_starts_app_with_supplied_args(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main parses CLI args and hands them to create_app + Flask.run."""
    groups_json, photos_dir, gold, stitched, crops = fixture_paths

    captured: dict[str, object] = {}

    def fake_run(host: str, port: int, debug: bool) -> None:
        captured["host"] = host
        captured["port"] = port
        captured["debug"] = debug

    monkeypatch.setattr(
        "sys.argv",
        [
            "review_server.py",
            "--port",
            "9999",
            "--host",
            "0.0.0.0",
            "--groups-json",
            str(groups_json),
            "--photos-dir",
            str(photos_dir),
            "--gold-groups-json",
            str(gold),
            "--stitched-root",
            str(stitched),
            "--crop-reviews-json",
            str(crops),
        ],
    )
    with patch("review_server.Flask.run", side_effect=fake_run):
        rc = main()
    assert rc == 0
    assert captured["port"] == 9999
    assert captured["host"] == "0.0.0.0"
