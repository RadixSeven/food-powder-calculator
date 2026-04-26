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
from review_server import create_app, main


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
    # Thumbnails are referenced by filename.
    assert "/thumb/a.jpg" in body


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
