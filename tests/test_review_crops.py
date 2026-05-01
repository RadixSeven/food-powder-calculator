"""Tests for scripts/_review_crops.py — the bbox/crop review view.

Real Flask test client, real on-disk JSON manifests, mocked nothing.
The crops view's job is plain file IO (read manifests, read/write
crop_reviews.json) plus rendering — easy to test end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from flask import Flask
from PIL import Image

from _review_crops import _build_render_groups, encode_crop_id, register


def _solid_image(
    path: Path, color: tuple[int, int, int] = (200, 50, 50)
) -> None:
    Image.new("RGB", (40, 40), color).save(path, "JPEG")


def _make_app(
    *,
    gold_groups_json: Path,
    stitched_root: Path,
    crop_reviews_json: Path,
) -> Flask:
    app = Flask(__name__)
    register(
        app,
        gold_groups_json=gold_groups_json,
        stitched_root=stitched_root,
        crop_reviews_json=crop_reviews_json,
    )
    return app


def _write_gold(path: Path, groups: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"groups": groups}))


def _write_manifest(
    stitched_root: Path,
    group_id: str,
    crops: list[dict[str, str]],
) -> None:
    group_dir = stitched_root / group_id
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "manifest.json").write_text(
        json.dumps({"group_id": group_id, "store": "MOM", "crops": crops})
    )


@pytest.fixture
def fixture_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path]:
    """One-group fixture with one front + one nutrition crop.

    Returns ``(gold, stitched_root, crop_reviews_json, crop_path,
    source_path)``. Both crop and source are real on-disk JPEGs so the
    thumbnail endpoint actually serves bytes.
    """
    repo_root = tmp_path / "repo"
    raw_dir = repo_root / "data" / "raw_photos"
    crops_dir = (
        repo_root / "data" / "stitched_panels" / "20260426_mom_001" / "crops"
    )
    raw_dir.mkdir(parents=True)
    crops_dir.mkdir(parents=True)
    source_path = raw_dir / "PXL_20260426_165737642.jpg"
    front_crop_path = (
        crops_dir / "PXL_20260426_165737642__front__0030_0000_1000_1000.jpg"
    )
    nutrition_crop_path = (
        crops_dir / "PXL_20260426_165809667__nutrition__0030_0000_1000_0770.jpg"
    )
    nutrition_source_path = raw_dir / "PXL_20260426_165809667.jpg"
    for p in (
        source_path,
        front_crop_path,
        nutrition_crop_path,
        nutrition_source_path,
    ):
        _solid_image(p)

    gold = tmp_path / "gold.json"
    _write_gold(
        gold,
        [{"id": "20260426_mom_001", "store": "MOM"}],
    )
    stitched_root = repo_root / "data" / "stitched_panels"
    _write_manifest(
        stitched_root,
        "20260426_mom_001",
        [
            {
                "role": "front",
                "crop": str(front_crop_path.relative_to(repo_root)),
                "source_photo": str(source_path.relative_to(repo_root)),
            },
            {
                "role": "nutrition",
                "crop": str(nutrition_crop_path.relative_to(repo_root)),
                "source_photo": str(
                    nutrition_source_path.relative_to(repo_root)
                ),
            },
        ],
    )
    crop_reviews_json = tmp_path / "crop_reviews.json"
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)
    return (
        gold,
        stitched_root,
        crop_reviews_json,
        front_crop_path,
        source_path,
    )


# ---------------------------------------------------------------------------
# encode_crop_id
# ---------------------------------------------------------------------------


def test_encode_crop_id_single_chars_for_first_36() -> None:
    """Indices 0–35 use 0-9, a-z — single chars per user request."""
    assert encode_crop_id(0) == "0"
    assert encode_crop_id(9) == "9"
    assert encode_crop_id(10) == "a"
    assert encode_crop_id(35) == "z"


def test_encode_crop_id_extends_to_multi_char_beyond_36() -> None:
    """Index 36 rolls over to two characters — natural base-36 extension."""
    assert encode_crop_id(36) == "10"
    assert encode_crop_id(37) == "11"
    assert encode_crop_id(36 * 2) == "20"
    assert encode_crop_id(36 * 36 - 1) == "zz"
    assert encode_crop_id(36 * 36) == "100"


def test_encode_crop_id_rejects_negative_index() -> None:
    with pytest.raises(ValueError):
        encode_crop_id(-1)


# ---------------------------------------------------------------------------
# index renders manifests
# ---------------------------------------------------------------------------


def test_index_renders_each_crop_with_assigned_id(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    gold, stitched_root, reviews, _, _ = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    body = app.test_client().get("/crops/").data.decode()
    assert "20260426_mom_001" in body
    # Both crops appear, each with its filename and the source filename.
    assert "PXL_20260426_165737642__front__" in body
    assert "PXL_20260426_165737642.jpg" in body
    assert "PXL_20260426_165809667__nutrition__" in body
    assert "PXL_20260426_165809667.jpg" in body
    # Short ids 0 (front) and 1 (nutrition) — no other ids in this group.
    assert '<div class="crop-id">0</div>' in body
    assert '<div class="crop-id">1</div>' in body


def test_index_legend_and_instructions_appear_at_top_and_bottom(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """The user wants the legend + instructions in both places so they
    don't have to scroll back when reviewing a long page.
    """
    gold, stitched_root, reviews, _, _ = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    body = app.test_client().get("/crops/").data.decode()
    # ✅/✂/↪/💬 legend keys appear twice (once per legend block).
    assert body.count("this crop has been reviewed") == 2
    assert body.count("info IS visible") == 2
    # Instructions appear twice too.
    assert body.count("How to use this page") == 1
    assert body.count("Reminder:") == 1


def test_index_warns_when_strict_role_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group whose manifest has no front/nutrition/price-tag crops
    surfaces a warning — the reviewer needs to know the pipeline is
    short on data, not just that the crops are bad.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)
    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "20260426_mom_005", "store": "MOM"}])
    stitched_root = repo_root / "stitched"
    _write_manifest(stitched_root, "20260426_mom_005", [])
    reviews = tmp_path / "crop_reviews.json"

    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    body = app.test_client().get("/crops/").data.decode()
    assert "missing front" in body
    assert "missing nutrition" in body
    assert "missing price-tag" in body


def test_index_filters_to_reviewed_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest entries whose role isn't in REVIEWED_ROLES (e.g.
    ingredients, other-label) don't render — the user only wants to
    review the strict roles right now.
    """
    repo_root = tmp_path / "repo"
    raw_dir = repo_root / "data" / "raw"
    crops_dir = repo_root / "data" / "crops"
    raw_dir.mkdir(parents=True)
    crops_dir.mkdir(parents=True)
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)
    src = raw_dir / "PXL_x.jpg"
    front_crop = crops_dir / "PXL_x__front__0_0_1_1.jpg"
    ingredients_crop = crops_dir / "PXL_x__ingredients__0_0_1_1.jpg"
    for p in (src, front_crop, ingredients_crop):
        _solid_image(p)

    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "g1", "store": "MOM"}])
    stitched_root = repo_root / "stitched"
    _write_manifest(
        stitched_root,
        "g1",
        [
            {
                "role": "front",
                "crop": str(front_crop.relative_to(repo_root)),
                "source_photo": str(src.relative_to(repo_root)),
            },
            {
                "role": "ingredients",
                "crop": str(ingredients_crop.relative_to(repo_root)),
                "source_photo": str(src.relative_to(repo_root)),
            },
        ],
    )
    reviews = tmp_path / "crop_reviews.json"

    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    body = app.test_client().get("/crops/").data.decode()
    assert "PXL_x__front__" in body
    assert "PXL_x__ingredients__" not in body


def test_index_handles_groups_without_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gold group with no manifest yet (pipeline hasn't run) renders
    as zero crops with the missing-role warnings — not a 500.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)
    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "g_unbuilt", "store": "MOM"}])
    stitched_root = repo_root / "stitched"
    stitched_root.mkdir()
    reviews = tmp_path / "crop_reviews.json"

    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    response = app.test_client().get("/crops/")
    assert response.status_code == 200
    body = response.data.decode()
    assert "g_unbuilt" in body
    assert "missing front" in body


def test_index_renders_form_state_from_crop_reviews(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """A pre-existing crop_reviews.json should drive the rendered form
    values + checkbox state, so reload-after-restart doesn't lose work.
    """
    gold, stitched_root, reviews, front_crop, _ = fixture_paths
    reviews.write_text(
        json.dumps(
            {
                "groups": {
                    "20260426_mom_001": {
                        "comment": "look here",
                        "crops": {
                            front_crop.name: {
                                "reviewed": True,
                                "info_excluded": True,
                                "info_available_in": "1",
                            }
                        },
                    }
                }
            }
        )
    )
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    body = app.test_client().get("/crops/").data.decode()
    assert 'data-server-checked="1"' in body  # reviewed AND excluded
    assert 'data-server-value="1"' in body  # info_available_in
    assert "look here" in body


# ---------------------------------------------------------------------------
# image serving
# ---------------------------------------------------------------------------


def test_thumb_serves_crop_and_source(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """Both /crops/thumb/crop/... and /crops/thumb/source/... resolve
    via the manifest entries to actual on-disk JPEGs.
    """
    gold, stitched_root, reviews, front_crop, source_path = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    client = app.test_client()
    crop_url = f"/crops/thumb/crop/20260426_mom_001/{front_crop.name}"
    source_url = f"/crops/thumb/source/20260426_mom_001/{source_path.name}"
    assert client.get(crop_url).status_code == 200
    assert client.get(source_url).status_code == 200


def test_raw_serves_crop_and_source_unchanged(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    gold, stitched_root, reviews, front_crop, source_path = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    client = app.test_client()
    response = client.get(f"/crops/raw/crop/20260426_mom_001/{front_crop.name}")
    assert response.status_code == 200
    assert response.data == front_crop.read_bytes()


def test_thumb_404_for_unknown_kind(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """The image route distinguishes 'crop' vs 'source' from the URL;
    any other kind is a 404 rather than serving the wrong image.
    """
    gold, stitched_root, reviews, front_crop, _ = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    response = app.test_client().get(
        f"/crops/thumb/wrong/20260426_mom_001/{front_crop.name}"
    )
    assert response.status_code == 404


def test_thumb_404_when_manifest_missing(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    gold, stitched_root, reviews, _, _ = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    response = app.test_client().get(
        "/crops/thumb/crop/no_such_group/whatever.jpg"
    )
    assert response.status_code == 404


def test_thumb_404_for_unknown_filename(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    gold, stitched_root, reviews, _, _ = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    response = app.test_client().get(
        "/crops/thumb/crop/20260426_mom_001/never_in_manifest.jpg"
    )
    assert response.status_code == 404


def test_thumb_404_for_too_few_url_segments(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    gold, stitched_root, reviews, _, _ = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    # Only two segments — missing the trailing filename. The path
    # converter takes the rest of the URL, so this would be "crop/<gid>"
    # without a filename.
    response = app.test_client().get("/crops/thumb/crop/20260426_mom_001")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST handlers
# ---------------------------------------------------------------------------


def test_post_crop_persists_review_state(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    gold, stitched_root, reviews, front_crop, _ = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    response = app.test_client().post(
        f"/crops/api/crop/20260426_mom_001/{front_crop.name}",
        json={
            "reviewed": True,
            "info_excluded": True,
            "info_available_in": "2,3",
        },
    )
    assert response.status_code == 200
    payload = json.loads(reviews.read_text())
    crop_state = payload["groups"]["20260426_mom_001"]["crops"][front_crop.name]
    assert crop_state["reviewed"] is True
    assert crop_state["info_excluded"] is True
    assert crop_state["info_available_in"] == "2,3"


def test_post_group_persists_comment(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    gold, stitched_root, reviews, _, _ = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    response = app.test_client().post(
        "/crops/api/group/20260426_mom_001",
        json={"comment": "look at the curved bottle"},
    )
    assert response.status_code == 200
    payload = json.loads(reviews.read_text())
    assert (
        payload["groups"]["20260426_mom_001"]["comment"]
        == "look at the curved bottle"
    )


def test_post_handles_missing_json_body(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """Both POST handlers default to empty/false on an empty body."""
    gold, stitched_root, reviews, front_crop, _ = fixture_paths
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    client = app.test_client()
    assert (
        client.post(
            f"/crops/api/crop/20260426_mom_001/{front_crop.name}"
        ).status_code
        == 200
    )
    assert client.post("/crops/api/group/20260426_mom_001").status_code == 200


def test_post_replaces_existing_state_in_place(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """A second POST overwrites the same crop's previous state — and
    leaves other crops + groups in the file untouched.
    """
    gold, stitched_root, reviews, front_crop, _ = fixture_paths
    reviews.write_text(
        json.dumps(
            {
                "groups": {
                    "other": {"comment": "untouched"},
                    "20260426_mom_001": {
                        "crops": {
                            front_crop.name: {
                                "reviewed": False,
                                "info_excluded": True,
                                "info_available_in": "old",
                            }
                        }
                    },
                }
            }
        )
    )
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    app.test_client().post(
        f"/crops/api/crop/20260426_mom_001/{front_crop.name}",
        json={
            "reviewed": True,
            "info_excluded": False,
            "info_available_in": "",
        },
    )
    payload = json.loads(reviews.read_text())
    crop_state = payload["groups"]["20260426_mom_001"]["crops"][front_crop.name]
    assert crop_state["reviewed"] is True
    assert crop_state["info_excluded"] is False
    assert crop_state["info_available_in"] == ""
    # The unrelated group entry is preserved.
    assert payload["groups"]["other"]["comment"] == "untouched"


# ---------------------------------------------------------------------------
# defensive: malformed inputs
# ---------------------------------------------------------------------------


def test_build_render_groups_handles_non_object_gold(
    tmp_path: Path,
) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text("[]")
    result = _build_render_groups(
        gold_groups_json=gold,
        stitched_root=tmp_path,
        crop_reviews_json=tmp_path / "missing.json",
    )
    assert result == []


def test_build_render_groups_handles_non_list_groups(tmp_path: Path) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"groups": "oops"}))
    result = _build_render_groups(
        gold_groups_json=gold,
        stitched_root=tmp_path,
        crop_reviews_json=tmp_path / "missing.json",
    )
    assert result == []


def test_build_render_groups_skips_non_dict_group_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-edit gold-groups data may have garbled entries; render the
    good ones instead of crashing the page.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "groups": [
                    "not a group",
                    {"id": 42, "store": "MOM"},  # id not str
                    {"id": "g_real", "store": "MOM"},
                ]
            }
        )
    )
    stitched_root = repo_root / "stitched"
    stitched_root.mkdir()
    result = _build_render_groups(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=tmp_path / "missing.json",
    )
    assert [g["id"] for g in result] == ["g_real"]


def test_build_render_groups_skips_malformed_manifest_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest entries with the wrong shape (missing keys, wrong
    types) get skipped instead of breaking the page.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)
    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "g1", "store": "MOM"}])
    stitched_root = repo_root / "stitched"
    group_dir = stitched_root / "g1"
    group_dir.mkdir(parents=True)
    (group_dir / "manifest.json").write_text(
        json.dumps(
            {
                "group_id": "g1",
                "store": "MOM",
                "crops": [
                    "not a dict",
                    {"role": "front"},  # missing crop + source_photo
                    {"role": 42, "crop": "x", "source_photo": "y"},
                    {"role": "front", "crop": 42, "source_photo": "y"},
                ],
            }
        )
    )
    result = _build_render_groups(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=tmp_path / "missing.json",
    )
    assert result[0]["crop_count"] == 0


def test_thumb_404_when_manifest_entry_has_non_string_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolver must skip manifest entries whose crop/source_photo
    fields aren't strings — otherwise it could 500 trying to Path(int).
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)
    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "g1", "store": "MOM"}])
    stitched_root = repo_root / "stitched"
    group_dir = stitched_root / "g1"
    group_dir.mkdir(parents=True)
    (group_dir / "manifest.json").write_text(
        json.dumps(
            {
                "crops": [
                    {"role": "front", "crop": 42, "source_photo": "x.jpg"},
                ]
            }
        )
    )
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=tmp_path / "missing.json",
    )
    response = app.test_client().get("/crops/thumb/crop/g1/x.jpg")
    assert response.status_code == 404


def test_index_handles_non_dict_review_state_for_a_group(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """A garbled crop_reviews.json (non-dict crops field) shouldn't
    crash; the page falls back to default form values.
    """
    gold, stitched_root, reviews, _, _ = fixture_paths
    reviews.write_text(
        json.dumps(
            {
                "groups": {
                    "20260426_mom_001": {
                        "comment": "ok",
                        "crops": "not a dict",
                    }
                }
            }
        )
    )
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    body = app.test_client().get("/crops/").data.decode()
    # No crash; default checkbox state is unchecked everywhere.
    assert 'data-server-checked="0"' in body


def test_load_crop_reviews_returns_empty_for_non_dict_groups_field(
    tmp_path: Path,
) -> None:
    """If groups isn't an object, the loader returns {} so the page
    doesn't try to iterate a string.
    """
    from _review_crops import _load_crop_reviews

    path = tmp_path / "crop_reviews.json"
    path.write_text(json.dumps({"groups": "oops"}))
    assert _load_crop_reviews(path) == {}


def test_thumb_returns_404_when_manifest_source_path_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest path that doesn't resolve is a data problem (typically
    a stale manifest from before photos moved into a batch subdirectory)
    and the resolver MUST return None rather than fall back to a glob
    search. The page-render path surfaces this as a visible error so
    the user sees the data problem instead of a broken-image icon.

    Regression test for a real bug on mom_002: an early version of the
    resolver grew a fallback that papered over the stale manifest,
    hiding the fact that group_pipeline.py needed to be re-run.
    """
    repo_root = tmp_path / "repo"
    raw_dir = repo_root / "data" / "raw_photos"
    batch_dir = raw_dir / "2026-04-26-shopping"
    crops_dir = repo_root / "data" / "stitched_panels" / "g1" / "crops"
    batch_dir.mkdir(parents=True)
    crops_dir.mkdir(parents=True)
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)

    # Source photo lives under the batch subdir (the new layout).
    _solid_image(batch_dir / "PXL_20260426_165855659.jpg")
    crop = crops_dir / "PXL_20260426_165855659__front__0_0_1_1.jpg"
    _solid_image(crop)

    # Manifest's source_photo points at the OLD path which no longer resolves.
    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "g1", "store": "MOM"}])
    stitched_root = repo_root / "data" / "stitched_panels"
    _write_manifest(
        stitched_root,
        "g1",
        [
            {
                "role": "front",
                "crop": str(crop.relative_to(repo_root)),
                "source_photo": "data/raw_photos/PXL_20260426_165855659.jpg",
            }
        ],
    )

    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=tmp_path / "missing.json",
    )
    # 404 from the route, NOT a found image via fallback.
    response = app.test_client().get(
        "/crops/raw/source/g1/PXL_20260426_165855659.jpg"
    )
    assert response.status_code == 404


def test_index_renders_visible_error_for_missing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a manifest's source path doesn't resolve, the page surfaces
    an explicit ``⚠ source missing`` block instead of a broken-image
    icon — so the data problem is visible at a glance and the user
    knows to refresh the manifest.
    """
    repo_root = tmp_path / "repo"
    raw_dir = repo_root / "data" / "raw_photos" / "batch"
    crops_dir = repo_root / "data" / "stitched_panels" / "g1" / "crops"
    raw_dir.mkdir(parents=True)
    crops_dir.mkdir(parents=True)
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)

    crop = crops_dir / "PXL_20260426_165855659__front__0_0_1_1.jpg"
    _solid_image(crop)
    # Source photo is intentionally NOT created at the recorded path.

    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "g1", "store": "MOM"}])
    stitched_root = repo_root / "data" / "stitched_panels"
    _write_manifest(
        stitched_root,
        "g1",
        [
            {
                "role": "front",
                "crop": str(crop.relative_to(repo_root)),
                "source_photo": "data/raw_photos/PXL_20260426_165855659.jpg",
            }
        ],
    )

    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=tmp_path / "missing.json",
    )
    body = app.test_client().get("/crops/").data.decode()
    assert "⚠ source missing" in body
    assert "re-run group_pipeline.py to refresh" in body
    # Crop is fine, so no crop-missing marker for this entry.
    assert "⚠ crop missing" not in body


def test_index_renders_visible_error_for_missing_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same shape as missing-source but for the crop side. Visible
    error rather than a broken-image icon, so the user knows the
    pipeline output (not the source photo) is the gap.
    """
    repo_root = tmp_path / "repo"
    raw_dir = repo_root / "data" / "raw_photos" / "batch"
    crops_dir = repo_root / "data" / "stitched_panels" / "g1" / "crops"
    raw_dir.mkdir(parents=True)
    crops_dir.mkdir(parents=True)
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)

    src = raw_dir / "PXL_20260426_165855659.jpg"
    _solid_image(src)
    # Crop intentionally NOT created at the recorded path.

    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "g1", "store": "MOM"}])
    stitched_root = repo_root / "data" / "stitched_panels"
    _write_manifest(
        stitched_root,
        "g1",
        [
            {
                "role": "front",
                "crop": "data/stitched_panels/g1/crops/missing-crop.jpg",
                "source_photo": str(src.relative_to(repo_root)),
            }
        ],
    )

    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=tmp_path / "missing.json",
    )
    body = app.test_client().get("/crops/").data.decode()
    assert "⚠ crop missing" in body
    assert "⚠ source missing" not in body


def test_thumb_404_when_manifest_crops_field_is_not_a_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve_image must return None when manifest.crops is the wrong
    shape — a 404 from the resolver becomes a 404 from the route.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)
    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "g1", "store": "MOM"}])
    stitched_root = repo_root / "stitched"
    group_dir = stitched_root / "g1"
    group_dir.mkdir(parents=True)
    (group_dir / "manifest.json").write_text(
        json.dumps({"crops": "not a list"})
    )
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=tmp_path / "missing.json",
    )
    response = app.test_client().get("/crops/thumb/crop/g1/anything.jpg")
    assert response.status_code == 404


def test_thumb_skips_non_dict_manifest_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a manifest entry is the wrong shape, the resolver skips it
    rather than crashing on .get of a non-dict.
    """
    repo_root = tmp_path / "repo"
    raw_dir = repo_root / "raw"
    raw_dir.mkdir(parents=True)
    src = raw_dir / "real.jpg"
    _solid_image(src)
    monkeypatch.setattr("_review_crops.REPO_ROOT", repo_root)
    gold = tmp_path / "gold.json"
    _write_gold(gold, [{"id": "g1", "store": "MOM"}])
    stitched_root = repo_root / "stitched"
    group_dir = stitched_root / "g1"
    group_dir.mkdir(parents=True)
    (group_dir / "manifest.json").write_text(
        json.dumps(
            {
                "crops": [
                    "not a dict",
                    {
                        "role": "front",
                        "crop": str(src.relative_to(repo_root)),
                        "source_photo": str(src.relative_to(repo_root)),
                    },
                ]
            }
        )
    )
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=tmp_path / "missing.json",
    )
    # The valid entry resolves; the bad one is silently skipped.
    response = app.test_client().get(f"/crops/thumb/crop/g1/{src.name}")
    assert response.status_code == 200


def test_ordered_manifest_entries_returns_empty_for_non_list_crops(
    tmp_path: Path,
) -> None:
    """The display-order helper falls back to [] when crops isn't a list,
    keeping the page rendering rather than 500-ing on garbled input.
    """
    from _review_crops import _ordered_manifest_entries

    assert _ordered_manifest_entries({"crops": "oops"}) == []
    assert _ordered_manifest_entries({}) == []


def test_post_handles_non_dict_existing_payload(
    fixture_paths: tuple[Path, Path, Path, Path, Path],
) -> None:
    """If crop_reviews.json was hand-edited to a non-object top-level
    state, the POST should still write a fresh shell rather than
    crash.
    """
    gold, stitched_root, reviews, front_crop, _ = fixture_paths
    reviews.write_text("[]")  # JSON array, not object
    app = _make_app(
        gold_groups_json=gold,
        stitched_root=stitched_root,
        crop_reviews_json=reviews,
    )
    response = app.test_client().post(
        f"/crops/api/crop/20260426_mom_001/{front_crop.name}",
        json={
            "reviewed": True,
            "info_excluded": False,
            "info_available_in": "",
        },
    )
    assert response.status_code == 200
    payload = json.loads(reviews.read_text())
    crop_state = payload["groups"]["20260426_mom_001"]["crops"][front_crop.name]
    assert crop_state["reviewed"] is True
