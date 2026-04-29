"""Tests for scripts/stitch.py.

The Claude calls are mocked everywhere; PIL operations run on real (small)
synthetic images.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from _claude import ClaudeStructuredOutputError
from find_legible_size import ExtractedPayload, SearchResult
from PIL import Image
from sizing_model import Probe
from stitch import (
    SizedPhoto,
    compute_min_legible_size,
    main,
    stitch_all_groups,
    stitch_group,
)


def _solid_image(
    path: Path, size: tuple[int, int], rgb: tuple[int, int, int]
) -> None:
    Image.new("RGB", size, rgb).save(path, "JPEG")


# ---------- compute_min_legible_size --------------------------------------


def test_compute_min_legible_size_returns_search_result_size(
    tmp_path: Path,
) -> None:
    photo = tmp_path / "p.jpg"
    _solid_image(photo, (200, 200), (255, 0, 0))

    captured: dict[str, object] = {}

    def fake_extract(
        p: Path, model: str, longest_side: int | None = None
    ) -> ExtractedPayload:
        return ExtractedPayload(text="t", barcodes=())

    def fake_select(
        _p: Path, _ref: ExtractedPayload, **_k: object
    ) -> tuple[str, ExtractedPayload]:
        return "haiku", ExtractedPayload(text="t", barcodes=())

    def fake_search(
        _p: Path, _model: str, _ref: ExtractedPayload, **k: object
    ) -> SearchResult:
        captured["photo_id"] = k["photo_id"]
        return SearchResult(
            photo_id=str(k["photo_id"]),
            reference_model="opus",
            search_model="haiku",
            min_legible_size=987,
            probes=[Probe(photo_id=str(k["photo_id"]), size=987, matched=True)],
        )

    with (
        patch("stitch.extract_payload", side_effect=fake_extract),
        patch("stitch.select_search_model", side_effect=fake_select),
        patch("stitch.binary_search_min_size", side_effect=fake_search),
    ):
        size = compute_min_legible_size(photo)
    assert size == 987
    assert captured["photo_id"] == "p.jpg"


# ---------- stitch_group --------------------------------------------------


def test_stitch_group_writes_jpeg_with_max_width_and_total_height(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    _solid_image(a, (100, 80), (255, 0, 0))
    _solid_image(b, (200, 60), (0, 255, 0))
    out = tmp_path / "out" / "group.jpg"

    stitch_group(
        "g1",
        [
            SizedPhoto(path=a, longest_side=100),
            SizedPhoto(path=b, longest_side=200),
        ],
        out,
        resize_cache_dir=tmp_path / "rs",
    )

    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (200, 80 + 4 + 60)


def test_stitch_group_handles_single_photo_without_separator(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a.jpg"
    _solid_image(a, (100, 80), (255, 0, 0))
    out = tmp_path / "out" / "g.jpg"
    stitch_group(
        "g",
        [SizedPhoto(path=a, longest_side=100)],
        out,
        resize_cache_dir=tmp_path / "rs",
    )
    with Image.open(out) as img:
        assert img.size == (100, 80)


def test_stitch_group_rejects_empty_input(tmp_path: Path) -> None:
    out = tmp_path / "g.jpg"
    with pytest.raises(ValueError, match="no photos"):
        stitch_group("g1", [], out, resize_cache_dir=tmp_path / "rs")


# ---------- stitch_all_groups --------------------------------------------


@pytest.fixture
def fake_gold(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a minimal groups JSON pointing at tmp photos. Returns
    (gold_groups_json, photos_dir, out_dir).
    """
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    out_dir = tmp_path / "out"
    p1 = photos_dir / "PXL_001.jpg"
    p2 = photos_dir / "PXL_002.jpg"
    _solid_image(p1, (120, 80), (255, 0, 0))
    _solid_image(p2, (60, 50), (0, 255, 0))
    gold_groups_json = tmp_path / "gold.json"
    gold_groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "g1",
                        "store": "MOM",
                        "photos": [
                            {
                                "path": str(p1.relative_to(tmp_path)),
                                "roles": ["front"],
                            },
                            {
                                "path": str(p2.relative_to(tmp_path)),
                                "roles": ["nutrition"],
                            },
                        ],
                        "warnings": [],
                    }
                ]
            }
        )
    )
    return gold_groups_json, photos_dir, out_dir


def test_stitch_all_groups_writes_one_jpeg_per_group(
    fake_gold: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold_groups_json, _photos_dir, out_dir = fake_gold
    monkeypatch.setattr("stitch.REPO_ROOT", tmp_path)
    monkeypatch.setattr("stitch.compute_min_legible_size", lambda _p: 100)
    monkeypatch.setattr("stitch.STITCH_RESIZE_CACHE", tmp_path / "stitch_rs")

    out_paths = stitch_all_groups(gold_groups_json, out_dir)
    assert len(out_paths) == 1
    assert out_paths[0].name == "g1.jpg"
    assert out_paths[0].exists()


def test_stitch_all_groups_skips_already_stitched_groups(
    fake_gold: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotency: re-running on a group whose stitched file exists is a no-op."""
    from unittest.mock import MagicMock

    gold_groups_json, _photos_dir, out_dir = fake_gold
    monkeypatch.setattr("stitch.REPO_ROOT", tmp_path)
    out_dir.mkdir(parents=True)
    (out_dir / "g1.jpg").write_bytes(b"existing")

    fake_compute = MagicMock(return_value=100)
    monkeypatch.setattr("stitch.compute_min_legible_size", fake_compute)
    out_paths = stitch_all_groups(gold_groups_json, out_dir)
    assert len(out_paths) == 1
    # Compute was NOT called — the group was skipped.
    fake_compute.assert_not_called()
    # And the existing file is untouched.
    assert (out_dir / "g1.jpg").read_bytes() == b"existing"


def test_stitch_all_groups_rejects_non_object_top_level(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        stitch_all_groups(bad, tmp_path)


def test_stitch_all_groups_rejects_missing_groups_field(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"oops": []}))
    with pytest.raises(ValueError, match="groups list"):
        stitch_all_groups(bad, tmp_path)


def test_stitch_all_groups_skips_photos_whose_sizing_fails(
    fake_gold: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If compute_min_legible_size raises ClaudeStructuredOutputError on a
    given photo (the reference Opus call couldn't satisfy the schema), that
    photo is dropped from the group and the remaining photos still get
    stitched.
    """
    gold_groups_json, _photos_dir, out_dir = fake_gold
    monkeypatch.setattr("stitch.REPO_ROOT", tmp_path)
    monkeypatch.setattr("stitch.STITCH_RESIZE_CACHE", tmp_path / "stitch_rs")

    failing_name = "PXL_002.jpg"

    def fake_compute(p: Path) -> int:
        if p.name == failing_name:
            raise ClaudeStructuredOutputError(
                returncode=1,
                stdout="",
                stderr="",
                cmd=[],
                failure_message="schema-invalid output after 5 retries",
            )
        return 100

    monkeypatch.setattr("stitch.compute_min_legible_size", fake_compute)
    out_paths = stitch_all_groups(gold_groups_json, out_dir)
    assert len(out_paths) == 1
    # The group still produced a stitched JPEG (with one photo dropped).
    assert out_paths[0].exists()


def test_stitch_all_groups_skips_group_when_every_photo_fails_sizing(
    fake_gold: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every photo in a group fails sizing, the group is skipped rather
    than producing an empty stitched image (or crashing on stitch_group's
    empty-input ValueError).
    """
    gold_groups_json, _photos_dir, out_dir = fake_gold
    monkeypatch.setattr("stitch.REPO_ROOT", tmp_path)
    monkeypatch.setattr("stitch.STITCH_RESIZE_CACHE", tmp_path / "stitch_rs")

    def always_fail(_p: Path) -> int:
        raise ClaudeStructuredOutputError(
            returncode=1,
            stdout="",
            stderr="",
            cmd=[],
            failure_message="schema-invalid output after 5 retries",
        )

    monkeypatch.setattr("stitch.compute_min_legible_size", always_fail)
    out_paths = stitch_all_groups(gold_groups_json, out_dir)
    assert out_paths == []
    # No stitched file was created for the all-failing group.
    assert not (out_dir / "g1.jpg").exists()


def test_stitch_all_groups_skips_non_dict_photo_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A photos array containing non-dict entries (a stray string) is
    tolerated — those entries are skipped, valid ones are still stitched.
    """
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    p = photos_dir / "PXL_001.jpg"
    _solid_image(p, (40, 40), (255, 0, 0))
    gold = tmp_path / "g.json"
    gold.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "real",
                        "store": "MOM",
                        "photos": [
                            "stray-string-not-a-dict",
                            {
                                "path": str(p.relative_to(tmp_path)),
                                "roles": ["front"],
                            },
                        ],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr("stitch.REPO_ROOT", tmp_path)
    monkeypatch.setattr("stitch.compute_min_legible_size", lambda _p: 40)
    monkeypatch.setattr("stitch.STITCH_RESIZE_CACHE", tmp_path / "stitch_rs")
    out_paths = stitch_all_groups(gold, tmp_path / "out")
    assert [pp.name for pp in out_paths] == ["real.jpg"]


def test_stitch_all_groups_skips_malformed_group_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Garbage in the groups array (non-dict, missing id, bad photos) is
    silently skipped — defensive against partial / hand-edited files.
    """
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    p = photos_dir / "PXL_001.jpg"
    _solid_image(p, (40, 40), (255, 0, 0))
    gold = tmp_path / "g.json"
    gold.write_text(
        json.dumps(
            {
                "groups": [
                    "not a dict",
                    {"store": "MOM", "photos": []},  # missing id
                    {"id": 42, "photos": []},  # id wrong type
                    {"id": "ok", "store": "MOM", "photos": "garbled"},
                    {
                        "id": "real",
                        "store": "MOM",
                        "photos": [
                            {
                                "path": str(p.relative_to(tmp_path)),
                                "roles": ["front"],
                            }
                        ],
                    },
                ]
            }
        )
    )
    monkeypatch.setattr("stitch.REPO_ROOT", tmp_path)
    monkeypatch.setattr("stitch.compute_min_legible_size", lambda _p: 40)
    monkeypatch.setattr("stitch.STITCH_RESIZE_CACHE", tmp_path / "stitch_rs")
    out_dir = tmp_path / "out"
    out_paths = stitch_all_groups(gold, out_dir)
    assert [p.name for p in out_paths] == ["real.jpg"]


def test_main_drives_stitch_all_groups(
    fake_gold: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gold_groups_json, _photos_dir, out_dir = fake_gold
    monkeypatch.setattr("stitch.REPO_ROOT", tmp_path)
    monkeypatch.setattr("stitch.compute_min_legible_size", lambda _p: 100)
    monkeypatch.setattr("stitch.STITCH_RESIZE_CACHE", tmp_path / "stitch_rs")
    monkeypatch.setattr(
        "sys.argv",
        [
            "stitch.py",
            "--gold-groups",
            str(gold_groups_json),
            "--out-dir",
            str(out_dir),
        ],
    )
    rc = main()
    assert rc == 0
    assert "DONE" in capsys.readouterr().err
