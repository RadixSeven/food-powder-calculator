"""Tests for scripts/_review_views.py — shared helpers for view modules."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask
from PIL import Image

from _review_views import load_json_object, register_image_routes


def test_load_json_object_returns_none_for_missing_file(tmp_path: Path) -> None:
    """Caller renders an empty UI rather than 500-ing."""
    assert load_json_object(tmp_path / "nope.json") is None


def test_load_json_object_returns_none_for_non_object_top_level(
    tmp_path: Path,
) -> None:
    """A JSON array (or any non-object) at the top level isn't a valid
    state file. Defensive narrowing keeps a malformed file from
    propagating Any deep into request handlers."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]))
    assert load_json_object(path) is None


def test_load_json_object_returns_dict_for_valid_object(tmp_path: Path) -> None:
    path = tmp_path / "good.json"
    path.write_text(json.dumps({"a": 1, "b": [2]}))
    assert load_json_object(path) == {"a": 1, "b": [2]}


def test_register_image_routes_serves_thumb_and_raw(tmp_path: Path) -> None:
    """The shared image route helper resolves the URL key via the
    caller-supplied resolver; thumb downscales while raw passes through."""
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (4000, 3000), (200, 50, 50)).save(img_path, "JPEG")

    def resolver(key: str) -> Path | None:
        return img_path if key == "the/key/here.jpg" else None

    app = Flask(__name__)
    register_image_routes(
        app, url_prefix="example", resolver=resolver, endpoint_prefix="ex"
    )
    client = app.test_client()
    raw = client.get("/example/raw/the/key/here.jpg")
    thumb = client.get("/example/thumb/the/key/here.jpg")
    assert raw.status_code == 200
    assert thumb.status_code == 200
    assert raw.data == img_path.read_bytes()
    # Thumbnail body should be smaller than raw body since it's
    # downscaled from 4000×3000 to ≤320 longest-side.
    assert len(thumb.data) < len(raw.data)


def test_register_image_routes_404_when_resolver_returns_none(
    tmp_path: Path,
) -> None:
    def resolver(key: str) -> Path | None:
        return None

    app = Flask(__name__)
    register_image_routes(
        app, url_prefix="example", resolver=resolver, endpoint_prefix="ex"
    )
    client = app.test_client()
    assert client.get("/example/thumb/anything.jpg").status_code == 404
    assert client.get("/example/raw/anything.jpg").status_code == 404


def test_register_image_routes_distinct_endpoints(tmp_path: Path) -> None:
    """Two views can each register thumb/raw routes on the same app
    without colliding in Flask's URL map; the endpoint_prefix arg keeps
    their endpoint names distinct."""
    img = tmp_path / "img.jpg"
    Image.new("RGB", (40, 40), (100, 100, 100)).save(img, "JPEG")

    def resolver(_key: str) -> Path:
        return img

    app = Flask(__name__)
    register_image_routes(
        app, url_prefix="a", resolver=resolver, endpoint_prefix="a"
    )
    register_image_routes(
        app, url_prefix="b", resolver=resolver, endpoint_prefix="b"
    )
    # Both work — no AssertionError on registration, both return 200.
    client = app.test_client()
    assert client.get("/a/thumb/x.jpg").status_code == 200
    assert client.get("/b/thumb/x.jpg").status_code == 200
