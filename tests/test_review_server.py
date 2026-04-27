"""Tests for scripts/review_server.py.

Uses Flask's test client; no real network. Image thumbnails are generated
on real (small) Pillow inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from review_server import _format_gap, create_app, main


def _solid_image(path: Path) -> None:
    Image.new("RGB", (200, 100), (200, 50, 50)).save(path, "JPEG")


@pytest.fixture
def fixture_paths(tmp_path: Path) -> tuple[Path, Path]:
    photos_dir = tmp_path / "raw"
    photos_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        _solid_image(photos_dir / name)
    groups_json = tmp_path / "groups.json"
    groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "g1",
                        "store": "MOM",
                        "photos": [
                            {"path": "raw/a.jpg", "roles": ["front"]},
                            {"path": "raw/b.jpg", "roles": ["nutrition"]},
                        ],
                        "warnings": ["missing shelf price tag"],
                        "locked": False,
                    },
                    {
                        "id": "g2",
                        "store": "CVS",
                        "photos": [
                            {"path": "raw/a.jpg", "roles": ["front"]},
                        ],
                        "warnings": [],
                        "locked": False,
                    },
                ]
            },
            indent=2,
        )
    )
    return groups_json, photos_dir


def test_index_renders_each_group(fixture_paths: tuple[Path, Path]) -> None:
    groups_json, photos_dir = fixture_paths
    app = create_app(groups_json=groups_json, photos_dir=photos_dir)
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    body = response.data.decode()
    assert "g1" in body
    assert "g2" in body
    assert "missing shelf price tag" in body
    # Thumbnails are referenced by filename and link to the raw photo.
    assert "/thumb/a.jpg" in body
    assert 'href="/raw/a.jpg"' in body
    # Filename is shown as a label so the user can scan the surrounding photos.
    assert ">a.jpg<" in body


def test_index_returns_empty_when_groups_json_missing(tmp_path: Path) -> None:
    """If no groups.json yet, render an empty list rather than 500."""
    app = create_app(groups_json=tmp_path / "missing.json", photos_dir=tmp_path)
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "0 groups" in response.data.decode()


def test_thumbnail_returns_jpeg(fixture_paths: tuple[Path, Path]) -> None:
    groups_json, photos_dir = fixture_paths
    app = create_app(groups_json=groups_json, photos_dir=photos_dir)
    client = app.test_client()
    response = client.get("/thumb/a.jpg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"


def test_thumbnail_404_for_unknown_filename(
    fixture_paths: tuple[Path, Path],
) -> None:
    groups_json, photos_dir = fixture_paths
    app = create_app(groups_json=groups_json, photos_dir=photos_dir)
    client = app.test_client()
    response = client.get("/thumb/nope.jpg")
    assert response.status_code == 404


def test_raw_returns_full_resolution_jpeg(
    fixture_paths: tuple[Path, Path],
) -> None:
    """Clicking a thumbnail opens /raw/<filename> in a new tab; serve it as-is."""
    groups_json, photos_dir = fixture_paths
    app = create_app(groups_json=groups_json, photos_dir=photos_dir)
    client = app.test_client()
    response = client.get("/raw/a.jpg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    # The /raw bytes match the on-disk file (no resize).
    assert response.data == (photos_dir / "a.jpg").read_bytes()


def test_raw_404_for_unknown_filename(
    fixture_paths: tuple[Path, Path],
) -> None:
    groups_json, photos_dir = fixture_paths
    app = create_app(groups_json=groups_json, photos_dir=photos_dir)
    client = app.test_client()
    response = client.get("/raw/nope.jpg")
    assert response.status_code == 404


def test_format_gap_seconds_is_short_when_under_threshold() -> None:
    text, short = _format_gap(
        "PXL_20260426_170617596.MP.jpg", "PXL_20260426_170636147.MP.jpg"
    )
    # 18.6 seconds → rendered as "19 s" and flagged short.
    assert text == "19 s"
    assert short is True


def test_format_gap_minutes() -> None:
    text, short = _format_gap(
        "PXL_20260426_165737000.jpg", "PXL_20260426_170012000.jpg"
    )
    # 2 min 35 s → not flagged short
    assert text == "2 min 35 s"
    assert short is False


def test_format_gap_hours() -> None:
    text, short = _format_gap(
        "PXL_20260426_170617596.MP.jpg", "PXL_20260426_180942709.MP.jpg"
    )
    # ~73 min → "1 h 13 min"
    assert text is not None and text.startswith("1 h ")
    assert short is False


def test_format_gap_returns_none_for_first_group() -> None:
    text, short = _format_gap(None, "PXL_20260426_170617596.MP.jpg")
    assert text is None
    assert short is False


def test_format_gap_returns_none_for_unparseable_names() -> None:
    text, short = _format_gap("foo.jpg", "bar.jpg")
    assert text is None
    assert short is False


def test_index_renders_inter_group_gaps(tmp_path: Path) -> None:
    """The index should display a time-delta header between groups."""
    photos_dir = tmp_path
    for name in (
        "PXL_20260426_170617596.MP.jpg",
        "PXL_20260426_170636147.MP.jpg",
        "PXL_20260426_180942709.MP.jpg",
    ):
        _solid_image(photos_dir / name)
    groups_json = tmp_path / "groups.json"
    groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "g1",
                        "store": "MOM",
                        "photos": [
                            {
                                "path": str(
                                    photos_dir / "PXL_20260426_170617596.MP.jpg"
                                ),
                                "roles": ["front"],
                            }
                        ],
                        "warnings": [],
                        "locked": False,
                    },
                    {
                        "id": "g2",
                        "store": "MOM",
                        "photos": [
                            {
                                "path": str(
                                    photos_dir / "PXL_20260426_170636147.MP.jpg"
                                ),
                                "roles": ["front"],
                            }
                        ],
                        "warnings": [],
                        "locked": False,
                    },
                    {
                        "id": "g3",
                        "store": "CVS",
                        "photos": [
                            {
                                "path": str(
                                    photos_dir / "PXL_20260426_180942709.MP.jpg"
                                ),
                                "roles": ["front"],
                            }
                        ],
                        "warnings": [],
                        "locked": False,
                    },
                ]
            }
        )
    )

    app = create_app(groups_json=groups_json, photos_dir=photos_dir)
    client = app.test_client()
    body = client.get("/").data.decode()
    # 19-second gap between g1 and g2 → flagged short.
    assert "19 s" in body
    assert 'class="gap short"' in body
    # ~1h gap between g2 and g3 → not flagged short.
    assert "1 h" in body


def test_thumb_resolves_extension_pool_paths(tmp_path: Path) -> None:
    """Photos that arrived via the boundary-resolution extension pool
    have paths that point outside ``photos_dir`` (e.g. data/raw_photos/).
    The server must still serve their thumbnails by following the path
    stored in groups.json."""
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    extension_dir = tmp_path / "extension"
    extension_dir.mkdir()
    _solid_image(sample_dir / "in_sample.jpg")
    _solid_image(extension_dir / "in_extension.jpg")
    groups_json = tmp_path / "groups.json"
    groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "g1",
                        "store": "MOM",
                        "photos": [
                            {
                                "path": str(sample_dir / "in_sample.jpg"),
                                "roles": ["front"],
                            },
                            {
                                "path": str(extension_dir / "in_extension.jpg"),
                                "roles": ["price-tag"],
                            },
                        ],
                        "warnings": [],
                        "locked": False,
                    },
                ]
            }
        )
    )

    app = create_app(groups_json=groups_json, photos_dir=sample_dir)
    client = app.test_client()
    # Sample photo: resolves via groups.json path AND via fallback dir.
    assert client.get("/thumb/in_sample.jpg").status_code == 200
    # Extension photo: resolves only via the path stored in groups.json.
    assert client.get("/thumb/in_extension.jpg").status_code == 200
    assert client.get("/raw/in_extension.jpg").status_code == 200


def test_thumb_resolves_repo_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paths stored as repo-relative (e.g. data/raw_photos/x.jpg) are
    resolved against REPO_ROOT."""
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    (fake_root / "data" / "raw_photos").mkdir(parents=True)
    _solid_image(fake_root / "data" / "raw_photos" / "rel.jpg")
    groups_json = tmp_path / "groups.json"
    groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "g1",
                        "store": "MOM",
                        "photos": [
                            {
                                "path": "data/raw_photos/rel.jpg",
                                "roles": ["front"],
                            }
                        ],
                        "warnings": [],
                        "locked": False,
                    }
                ]
            }
        )
    )

    monkeypatch.setattr("review_server.REPO_ROOT", fake_root)
    app = create_app(groups_json=groups_json, photos_dir=tmp_path)
    client = app.test_client()
    assert client.get("/thumb/rel.jpg").status_code == 200


def test_post_updates_group_in_place(fixture_paths: tuple[Path, Path]) -> None:
    groups_json, photos_dir = fixture_paths
    app = create_app(groups_json=groups_json, photos_dir=photos_dir)
    client = app.test_client()
    response = client.post(
        "/api/group/g1",
        json={"has_errors": True, "comment": "boundary off by one"},
    )
    assert response.status_code == 200

    payload = json.loads(groups_json.read_text())
    g1 = next(g for g in payload["groups"] if g["id"] == "g1")
    assert g1["has_errors"] is True
    assert g1["comment"] == "boundary off by one"


def test_post_404_when_group_id_not_found(
    fixture_paths: tuple[Path, Path],
) -> None:
    groups_json, photos_dir = fixture_paths
    app = create_app(groups_json=groups_json, photos_dir=photos_dir)
    client = app.test_client()
    response = client.post(
        "/api/group/no-such-id",
        json={"has_errors": True, "comment": ""},
    )
    assert response.status_code == 404


def test_post_handles_missing_json_body(
    fixture_paths: tuple[Path, Path],
) -> None:
    """A POST with no body should not crash; defaults to errors=False, no comment."""
    groups_json, photos_dir = fixture_paths
    app = create_app(groups_json=groups_json, photos_dir=photos_dir)
    client = app.test_client()
    response = client.post("/api/group/g1")
    assert response.status_code == 200

    payload = json.loads(groups_json.read_text())
    g1 = next(g for g in payload["groups"] if g["id"] == "g1")
    assert g1["has_errors"] is False
    assert g1["comment"] == ""


def test_main_starts_app_with_supplied_args(
    fixture_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Main parses CLI args and hands them to create_app + Flask.run."""
    groups_json, photos_dir = fixture_paths

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
        ],
    )
    with patch("review_server.Flask.run", side_effect=fake_run):
        rc = main()
    assert rc == 0
    assert captured["port"] == 9999
    assert captured["host"] == "0.0.0.0"
