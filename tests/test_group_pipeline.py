"""Tests for scripts/group_pipeline.py — orchestrator stitching strict roles."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from _json_types import JsonValue
import pytest

from group_pipeline import (
    CropResult,
    assign_expected_roles,
    detect_and_crop_group,
    load_group,
    process_group,
    stitch_role_outputs,
    write_manifest,
)
from role_bbox import PanelBbox


def _save(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (200, 100, 50)).save(path, "JPEG", quality=85)


def _bbox(kind: str, text_direction: str = "horizontal") -> PanelBbox:
    return PanelBbox(
        kind=kind,
        x_min_frac=0.0,
        y_min_frac=0.0,
        x_max_frac=1.0,
        y_max_frac=1.0,
        text_direction=text_direction,
    )


# ---------------------------------------------------------------------------
# assign_expected_roles
# ---------------------------------------------------------------------------


def test_assign_expected_roles_filters_to_known_kinds() -> None:
    """Loose roles are valid kinds too; only completely unknown labels
    get filtered. The order is canonical (ROLE_KINDS) for prompt
    determinism across runs.
    """
    assert assign_expected_roles({"front", "nutrition", "made-up-role"}) == (
        "front",
        "nutrition",
    )


def test_assign_expected_roles_returns_empty_for_unknown_only() -> None:
    assert assign_expected_roles({"some-unknown"}) == ()


# ---------------------------------------------------------------------------
# stitch_role_outputs — single-shot bypass
# ---------------------------------------------------------------------------


def _crop(path: Path, kind: str, *, src: Path | None = None) -> CropResult:
    """Build a CropResult with the bbox covering the whole crop, defaulting
    ``source_photo`` to the crop itself for tests that don't track sources.
    """
    return CropResult(
        crop_path=path, source_photo=src or path, bbox=_bbox(kind)
    )


def test_single_shot_role_uses_crop_directly_without_stitching(
    tmp_path: Path,
) -> None:
    """The user's optimization: when a role has only one crop in the
    group, stitching adds no value (no other frame to align with).
    Hand the crop straight to downstream extraction.
    """
    crop = tmp_path / "single.jpg"
    _save(crop, (800, 600))
    by_role = {"nutrition": [_crop(crop, "nutrition")]}
    artifacts = stitch_role_outputs("g1", by_role, out_dir=tmp_path / "out")
    assert artifacts.panels_by_role["nutrition"] == crop
    assert artifacts.crops_by_role["nutrition"] == (crop,)
    assert artifacts.crop_sources[crop] == crop
    assert artifacts.stitched["nutrition"] is False


def test_multi_shot_role_stitches_along_majority_axis(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _save(a, (200, 100))
    _save(b, (200, 100))
    by_role = {
        "nutrition": [_crop(a, "nutrition"), _crop(b, "nutrition")],
    }
    artifacts = stitch_role_outputs("g1", by_role, out_dir=tmp_path / "out")
    out_path = artifacts.panels_by_role["nutrition"]
    assert out_path != a and out_path != b  # newly created stitched file
    assert artifacts.crops_by_role["nutrition"] == (a, b)
    assert artifacts.stitched["nutrition"] is True
    with Image.open(out_path) as result:
        with Image.open(a) as src:
            # Horizontal stitch: width = a.width + sep + b.width = 404
            assert result.size[0] > src.size[0]


def test_store_assigned_from_group_id_prefix(tmp_path: Path) -> None:
    crop = tmp_path / "x.jpg"
    _save(crop, (100, 100))
    by_role = {"front": [_crop(crop, "front")]}
    mom = stitch_role_outputs(
        "20260426_mom_007", by_role, out_dir=tmp_path / "out"
    )
    cvs = stitch_role_outputs(
        "20260426_cvs_002", by_role, out_dir=tmp_path / "out2"
    )
    other = stitch_role_outputs(
        "weirdly_named_group", by_role, out_dir=tmp_path / "out3"
    )
    assert mom.store == "MOM"
    assert cvs.store == "CVS"
    assert other.store == "UNKNOWN"


# ---------------------------------------------------------------------------
# process_group — end-to-end with mocked detect_panels
# ---------------------------------------------------------------------------


def test_process_group_skips_photos_with_no_known_roles(
    tmp_path: Path,
) -> None:
    """Photos whose gold roles list is empty (or all unknown) get
    skipped without calling the detector — they're loose-only and
    we have no constraint to feed in.
    """
    img = tmp_path / "p.jpg"
    _save(img, (1000, 1000))
    empty_roles: list[JsonValue] = []
    group: dict[str, JsonValue] = {
        "id": "test_group",
        "photos": [
            {"path": str(img.relative_to(tmp_path)), "roles": empty_roles},
        ],
    }
    with patch("group_pipeline.REPO_ROOT", tmp_path):
        with patch("group_pipeline.detect_panels") as mock_detect:
            artifacts = process_group(group, out_root=tmp_path / "out")
    mock_detect.assert_not_called()
    assert artifacts.panels_by_role == {}


def test_detect_and_crop_group_handles_malformed_photos(
    tmp_path: Path,
) -> None:
    """A non-list 'photos' field, non-dict photo entries, missing 'path',
    and non-list 'roles' all get skipped without raising — gold data is
    typed JsonObject but defensiveness catches mid-edit weirdness.
    """
    img = tmp_path / "good.jpg"
    _save(img, (100, 100))
    group: dict[str, JsonValue] = {
        "id": "g",
        "photos": [
            "not a dict",  # gets skipped
            {"path": 12345},  # path not a string, skipped
            {"path": str(img.relative_to(tmp_path)), "roles": "not a list"},
            # The above row gets roles=set() (empty) → expected=() → skipped
        ],
    }
    with patch("group_pipeline.REPO_ROOT", tmp_path):
        with patch("group_pipeline.detect_panels") as mock_detect:
            result = detect_and_crop_group(group, crop_dir=tmp_path / "crops")
    mock_detect.assert_not_called()
    assert result == {}


def test_detect_and_crop_group_handles_non_list_photos(
    tmp_path: Path,
) -> None:
    """If 'photos' isn't a list, the function returns the empty defaultdict
    without crashing — defensive guard matters because gold_groups.json
    is hand-edited.
    """
    group: dict[str, JsonValue] = {"id": "g", "photos": "wrong type"}
    with patch("group_pipeline.detect_panels") as mock_detect:
        result = detect_and_crop_group(group, crop_dir=tmp_path / "crops")
    mock_detect.assert_not_called()
    assert dict(result) == {}


def test_process_group_warns_when_detector_returns_no_panels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the detector returns () for a photo whose gold expects a role,
    log a warning and continue — losing a single photo shouldn't fail
    the whole group.
    """
    img = tmp_path / "p.jpg"
    _save(img, (100, 100))
    group: dict[str, JsonValue] = {
        "id": "g",
        "photos": [
            {"path": str(img.relative_to(tmp_path)), "roles": ["nutrition"]}
        ],
    }
    with patch("group_pipeline.REPO_ROOT", tmp_path):
        with patch("group_pipeline.detect_panels", return_value=()):
            artifacts = process_group(group, out_root=tmp_path / "out")
    err = capsys.readouterr().err
    assert "WARN" in err
    assert artifacts.panels_by_role == {}


def test_stitch_role_outputs_skips_empty_role_lists(tmp_path: Path) -> None:
    """A role key with an empty list of items shouldn't produce an output
    or crash — defensive guard against upstream ordering bugs.
    """
    artifacts = stitch_role_outputs(
        "g1", {"nutrition": []}, out_dir=tmp_path / "out"
    )
    assert artifacts.panels_by_role == {}


def test_process_group_rejects_missing_id(tmp_path: Path) -> None:
    group: dict[str, JsonValue] = {"photos": []}
    with pytest.raises(ValueError, match="missing string 'id'"):
        process_group(group, out_root=tmp_path / "out")


def test_write_manifest_records_crops_and_sources(tmp_path: Path) -> None:
    """The manifest is the canonical 'current crops' list — review tooling
    reads it instead of glob-ing the per-group crops/ directory which
    accumulates stale files as the bbox prompt evolves.
    """
    crop = tmp_path / "crops" / "p1__nutrition__0_0_1000_1000.jpg"
    src = tmp_path / "raw_photos" / "p1.jpg"
    crop.parent.mkdir()
    src.parent.mkdir()
    _save(crop, (100, 100))
    _save(src, (200, 200))
    artifacts = stitch_role_outputs(
        "20260426_mom_001",
        {"nutrition": [_crop(crop, "nutrition", src=src)]},
        out_dir=tmp_path / "out",
    )
    manifest_path = tmp_path / "out" / "manifest.json"
    write_manifest(artifacts, manifest_path)
    payload = json.loads(manifest_path.read_text())
    assert payload["group_id"] == "20260426_mom_001"
    assert payload["store"] == "MOM"
    assert payload["crops"] == [
        {
            "role": "nutrition",
            "crop": str(crop),
            "source_photo": str(src),
        }
    ]


def test_write_manifest_resolves_repo_relative_paths(tmp_path: Path) -> None:
    """Crops under REPO_ROOT serialize as repo-relative strings so the
    manifest is portable across machines / checkouts.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    crop = repo / "data" / "crops" / "p__front__0_0_1000_1000.jpg"
    src = repo / "data" / "raw" / "p.jpg"
    crop.parent.mkdir(parents=True)
    src.parent.mkdir(parents=True)
    _save(crop, (100, 100))
    _save(src, (100, 100))
    artifacts = stitch_role_outputs(
        "20260426_mom_001",
        {"front": [_crop(crop, "front", src=src)]},
        out_dir=repo / "out",
    )
    manifest_path = repo / "out" / "manifest.json"
    with patch("group_pipeline.REPO_ROOT", repo):
        write_manifest(artifacts, manifest_path)
    payload = json.loads(manifest_path.read_text())
    assert (
        payload["crops"][0]["crop"] == "data/crops/p__front__0_0_1000_1000.jpg"
    )
    assert payload["crops"][0]["source_photo"] == "data/raw/p.jpg"


def test_load_group_finds_matching_group(tmp_path: Path) -> None:
    gold = tmp_path / "g.json"
    gold.write_text(
        '{"groups": [{"id": "first", "photos": []}, '
        '{"id": "second", "photos": []}]}'
    )
    g = load_group(gold, "second")
    assert g["id"] == "second"


def test_load_group_raises_for_missing_group(tmp_path: Path) -> None:
    gold = tmp_path / "g.json"
    gold.write_text('{"groups": [{"id": "first", "photos": []}]}')
    with pytest.raises(KeyError, match="not in"):
        load_group(gold, "nonexistent")


def test_load_group_raises_for_non_object_root(tmp_path: Path) -> None:
    gold = tmp_path / "g.json"
    gold.write_text("[]")
    with pytest.raises(ValueError, match="Expected JSON object"):
        load_group(gold, "x")


def test_load_group_raises_for_missing_groups_key(tmp_path: Path) -> None:
    gold = tmp_path / "g.json"
    gold.write_text("{}")
    with pytest.raises(ValueError, match="Expected 'groups' list"):
        load_group(gold, "x")


def test_process_group_dispatches_expected_roles_per_photo(
    tmp_path: Path,
) -> None:
    """Each photo's bbox call should be parameterized with its own
    gold-derived expected_roles — this is what enables the schema-
    restriction to suppress off-target detections.
    """
    img1 = tmp_path / "p1.jpg"
    img2 = tmp_path / "p2.jpg"
    _save(img1, (1000, 1000))
    _save(img2, (1000, 1000))

    front_panel = _bbox("front")
    price_panel = _bbox("price-tag")

    def fake_detect(path: Path, **kw: object) -> tuple[PanelBbox, ...]:
        if path == img1:
            return (front_panel,)
        return (price_panel,)

    group: dict[str, JsonValue] = {
        "id": "test_group",
        "photos": [
            {"path": str(img1.relative_to(tmp_path)), "roles": ["front"]},
            {
                "path": str(img2.relative_to(tmp_path)),
                "roles": ["price-tag"],
            },
        ],
    }
    captured: list[tuple[str, tuple[str, ...]]] = []

    def recording_detect(path: Path, **kw: object) -> tuple[PanelBbox, ...]:
        roles = kw["expected_roles"]
        assert isinstance(roles, tuple)
        captured.append((path.name, roles))
        return fake_detect(path, **kw)

    with patch("group_pipeline.REPO_ROOT", tmp_path):
        with patch(
            "group_pipeline.detect_panels", side_effect=recording_detect
        ):
            with patch(
                "group_pipeline.crop_panel",
                side_effect=lambda src, panel, **kw: src,
            ):
                process_group(group, out_root=tmp_path / "out")
    assert captured == [
        ("p1.jpg", ("front",)),
        ("p2.jpg", ("price-tag",)),
    ]
