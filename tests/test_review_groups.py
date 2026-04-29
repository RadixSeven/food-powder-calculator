"""Tests for scripts/_review_groups.py — the groups review view.

Uses Flask's test client; no real network. Image thumbnails are
generated on real (small) Pillow inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from flask import Flask
from PIL import Image

from _review_groups import (
    _load_groups,
    _update_group_in_place,
    format_gap,
    register,
)


def _solid_image(path: Path) -> None:
    Image.new("RGB", (200, 100), (200, 50, 50)).save(path, "JPEG")


def _make_app(groups_json: Path, photos_dir: Path) -> Flask:
    """Build a Flask app with only the groups view registered.

    Tests should keep this in one place so a future signature change on
    :func:`_review_groups.register` only updates one helper.
    """
    app = Flask(__name__)
    register(app, groups_json=groups_json, photos_dir=photos_dir)
    return app


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
    app = _make_app(groups_json, photos_dir)
    response = app.test_client().get("/groups/")
    assert response.status_code == 200
    body = response.data.decode()
    assert "g1" in body
    assert "g2" in body
    assert "missing shelf price tag" in body
    # Thumbnails are referenced by filename and link to the raw photo.
    assert "/groups/thumb/a.jpg" in body
    assert 'href="/groups/raw/a.jpg"' in body
    # Filename is shown as a label so the user can scan the surrounding photos.
    assert ">a.jpg<" in body


def test_form_controls_carry_server_state_and_disable_autocomplete(
    fixture_paths: tuple[Path, Path],
) -> None:
    """Browsers' bfcache restores form values across page reloads, overriding
    server-rendered state. Defenses: ``autocomplete="off"`` on each control
    plus ``data-server-*`` attributes that JS re-applies on ``pageshow``.
    """
    groups_json, photos_dir = fixture_paths
    payload = json.loads(groups_json.read_text())
    g1 = next(g for g in payload["groups"] if g["id"] == "g1")
    g1["has_errors"] = True
    g1["comment"] = "needs another look"
    groups_json.write_text(json.dumps(payload))

    app = _make_app(groups_json, photos_dir)
    body = app.test_client().get("/groups/").data.decode()

    # Both controls disable browser autocomplete and carry server state in
    # data-server-* attributes that the page-load JS re-applies.
    assert 'class="errors-flag" autocomplete="off"' in body
    assert 'data-server-checked="1"' in body  # g1 is flagged
    assert 'data-server-checked="0"' in body  # g2 is not flagged
    assert (
        'class="comment" placeholder="comment (optional)" autocomplete="off"'
        in body
    )
    assert 'data-server-value="needs another look"' in body
    # The pageshow handler is what neutralizes bfcache restoration; assert
    # it's wired up so a future refactor can't drop it silently.
    assert "addEventListener('pageshow', syncFromServerState)" in body


def test_index_returns_empty_when_groups_json_missing(tmp_path: Path) -> None:
    """If no groups.json yet, render an empty list rather than 500."""
    app = _make_app(tmp_path / "missing.json", tmp_path)
    body = app.test_client().get("/groups/").data.decode()
    assert "0 groups" in body


def test_thumbnail_returns_jpeg(fixture_paths: tuple[Path, Path]) -> None:
    groups_json, photos_dir = fixture_paths
    app = _make_app(groups_json, photos_dir)
    response = app.test_client().get("/groups/thumb/a.jpg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"


def test_thumbnail_404_for_unknown_filename(
    fixture_paths: tuple[Path, Path],
) -> None:
    groups_json, photos_dir = fixture_paths
    app = _make_app(groups_json, photos_dir)
    response = app.test_client().get("/groups/thumb/nope.jpg")
    assert response.status_code == 404


def test_raw_returns_full_resolution_jpeg(
    fixture_paths: tuple[Path, Path],
) -> None:
    """Clicking a thumbnail opens /groups/raw/<filename> in a new tab."""
    groups_json, photos_dir = fixture_paths
    app = _make_app(groups_json, photos_dir)
    response = app.test_client().get("/groups/raw/a.jpg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data == (photos_dir / "a.jpg").read_bytes()


def test_raw_404_for_unknown_filename(
    fixture_paths: tuple[Path, Path],
) -> None:
    groups_json, photos_dir = fixture_paths
    app = _make_app(groups_json, photos_dir)
    response = app.test_client().get("/groups/raw/nope.jpg")
    assert response.status_code == 404


def test_format_gap_seconds_is_short_when_under_threshold() -> None:
    text, short = format_gap(
        "PXL_20260426_170617596.MP.jpg", "PXL_20260426_170636147.MP.jpg"
    )
    # 18.6 seconds → rendered as "19 s" and flagged short.
    assert text == "19 s"
    assert short is True


def test_format_gap_minutes() -> None:
    text, short = format_gap(
        "PXL_20260426_165737000.jpg", "PXL_20260426_170012000.jpg"
    )
    assert text == "2 min 35 s"
    assert short is False


def test_format_gap_hours() -> None:
    text, short = format_gap(
        "PXL_20260426_170617596.MP.jpg", "PXL_20260426_180942709.MP.jpg"
    )
    assert text is not None and text.startswith("1 h ")
    assert short is False


def test_format_gap_returns_none_for_first_group() -> None:
    text, short = format_gap(None, "PXL_20260426_170617596.MP.jpg")
    assert text is None
    assert short is False


def test_format_gap_returns_none_for_unparseable_names() -> None:
    text, short = format_gap("foo.jpg", "bar.jpg")
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

    app = _make_app(groups_json, photos_dir)
    body = app.test_client().get("/groups/").data.decode()
    # 19-second gap between g1 and g2 → flagged short.
    assert "19 s" in body
    assert 'class="gap short"' in body
    # ~1h gap between g2 and g3 → not flagged short.
    assert "1 h" in body


def test_load_groups_returns_empty_for_non_object_top_level(
    tmp_path: Path,
) -> None:
    groups_json = tmp_path / "g.json"
    groups_json.write_text("[]")
    assert _load_groups(groups_json) == []


def test_load_groups_returns_empty_when_groups_field_is_not_a_list(
    tmp_path: Path,
) -> None:
    groups_json = tmp_path / "g.json"
    groups_json.write_text(json.dumps({"groups": "oops"}))
    assert _load_groups(groups_json) == []


def test_load_groups_filters_out_non_dict_entries(tmp_path: Path) -> None:
    groups_json = tmp_path / "g.json"
    groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    "not a group",
                    {
                        "id": "real",
                        "store": "MOM",
                        "photos": [],
                        "warnings": [],
                    },
                ]
            }
        )
    )
    app = _make_app(groups_json, tmp_path)
    body = app.test_client().get("/groups/").data.decode()
    assert "1 groups" in body
    assert "real" in body


def test_thumb_skips_groups_with_non_list_photos_field(
    tmp_path: Path,
) -> None:
    """resolve_photo's iteration over groups must tolerate a malformed
    photos field (str instead of list) and skip non-dict entries instead of
    blowing up.
    """
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    _solid_image(photos_dir / "real.jpg")
    groups_json = tmp_path / "g.json"
    groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "g1",
                        "store": "MOM",
                        "photos": "oops",
                        "warnings": [],
                    },
                    {
                        "id": "g2",
                        "store": "MOM",
                        "photos": [
                            "just a string",
                            {
                                "path": str(photos_dir / "real.jpg"),
                                "roles": ["front"],
                            },
                        ],
                        "warnings": [],
                    },
                ]
            }
        )
    )
    app = _make_app(groups_json, photos_dir)
    response = app.test_client().get("/groups/thumb/real.jpg")
    assert response.status_code == 200


def test_index_tolerates_garbled_photo_entries(tmp_path: Path) -> None:
    """A group whose photos field isn't a list, or contains non-dict items
    or non-string paths, renders as if it had no photos rather than 500.
    """
    groups_json = tmp_path / "g.json"
    groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "g1",
                        "store": "MOM",
                        "photos": "oops",
                        "warnings": [],
                    },
                    {
                        "id": "g2",
                        "store": "MOM",
                        "photos": [
                            "not a dict",
                            {"path": 42, "roles": []},
                            {"path": "ok.jpg", "roles": ["front"]},
                        ],
                        "warnings": [],
                    },
                ]
            }
        )
    )
    app = _make_app(groups_json, tmp_path)
    assert app.test_client().get("/groups/").status_code == 200


def test_update_group_in_place_returns_false_for_non_object_top_level(
    tmp_path: Path,
) -> None:
    groups_json = tmp_path / "g.json"
    groups_json.write_text("[]")
    assert _update_group_in_place(groups_json, "any", True, "x") is False


def test_update_group_in_place_returns_false_when_groups_not_a_list(
    tmp_path: Path,
) -> None:
    groups_json = tmp_path / "g.json"
    groups_json.write_text(json.dumps({"groups": "oops"}))
    assert _update_group_in_place(groups_json, "any", True, "x") is False


def test_update_group_in_place_skips_non_dict_entries(tmp_path: Path) -> None:
    groups_json = tmp_path / "g.json"
    groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    "skip me",
                    {
                        "id": "real",
                        "store": "MOM",
                        "photos": [],
                        "warnings": [],
                    },
                ]
            }
        )
    )
    assert _update_group_in_place(groups_json, "real", True, "ok") is True


def test_thumb_resolves_extension_pool_paths(tmp_path: Path) -> None:
    """Photos that arrived via the boundary-resolution extension pool
    have paths that point outside ``photos_dir`` (e.g. data/raw_photos/).
    The server must still serve their thumbnails by following the path
    stored in groups.json.
    """
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

    app = _make_app(groups_json, sample_dir)
    client = app.test_client()
    # Sample photo: resolves via groups.json path AND via fallback dir.
    assert client.get("/groups/thumb/in_sample.jpg").status_code == 200
    # Extension photo: resolves only via the path stored in groups.json.
    assert client.get("/groups/thumb/in_extension.jpg").status_code == 200
    assert client.get("/groups/raw/in_extension.jpg").status_code == 200


def test_thumb_resolves_repo_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paths stored as repo-relative (e.g. data/raw_photos/x.jpg) are
    resolved against REPO_ROOT.
    """
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

    monkeypatch.setattr("_review_groups.REPO_ROOT", fake_root)
    app = _make_app(groups_json, tmp_path)
    assert app.test_client().get("/groups/thumb/rel.jpg").status_code == 200


def test_post_updates_group_in_place(fixture_paths: tuple[Path, Path]) -> None:
    groups_json, photos_dir = fixture_paths
    app = _make_app(groups_json, photos_dir)
    response = app.test_client().post(
        "/groups/api/g1",
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
    app = _make_app(groups_json, photos_dir)
    response = app.test_client().post(
        "/groups/api/no-such-id",
        json={"has_errors": True, "comment": ""},
    )
    assert response.status_code == 404


def test_post_handles_missing_json_body(
    fixture_paths: tuple[Path, Path],
) -> None:
    """A POST with no body should not crash; defaults to errors=False."""
    groups_json, photos_dir = fixture_paths
    app = _make_app(groups_json, photos_dir)
    response = app.test_client().post("/groups/api/g1")
    assert response.status_code == 200

    payload = json.loads(groups_json.read_text())
    g1 = next(g for g in payload["groups"] if g["id"] == "g1")
    assert g1["has_errors"] is False
    assert g1["comment"] == ""
