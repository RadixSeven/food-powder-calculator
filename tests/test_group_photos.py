"""Tests for scripts/group_photos.py.

The Sonnet calls are mocked everywhere; image resizing uses real Pillow on
small synthetic images.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from group_photos import (
    CVS_CUTOFF_FILENAME,
    Group,
    GROUPING_JSON_SCHEMA,
    GROUPING_SYSTEM_PROMPT,
    PhotoEntry,
    VALID_ROLES,
    _extract_json_object,
    _parse_classification,
    assign_groups,
    classify_photo,
    compute_group_warnings,
    list_photos,
    main,
    make_group_id,
    store_for,
    write_groups,
)
from PIL import Image


def _solid_image(path: Path) -> None:
    Image.new("RGB", (200, 100), (200, 50, 50)).save(path, "JPEG")


# ---------- pure helpers ------------------------------------------------------


def test_store_for_returns_mom_before_cutoff() -> None:
    assert store_for("PXL_20260426_165737642.jpg") == "MOM"


def test_store_for_returns_cvs_at_or_after_cutoff() -> None:
    assert store_for(CVS_CUTOFF_FILENAME) == "CVS"
    assert store_for("PXL_20260426_181000000.jpg") == "CVS"


def test_make_group_id_format() -> None:
    assert make_group_id("MOM", 1) == "20260426_mom_001"
    assert make_group_id("CVS", 42) == "20260426_cvs_042"


def test_compute_group_warnings_all_present() -> None:
    g = Group(
        id="x",
        store="MOM",
        photos=[
            PhotoEntry(path="a", roles=["front"]),
            PhotoEntry(path="b", roles=["nutrition", "ingredients"]),
            PhotoEntry(path="c", roles=["price-tag"]),
        ],
    )
    assert compute_group_warnings(g) == []


def test_compute_group_warnings_flags_each_missing() -> None:
    g = Group(
        id="x",
        store="MOM",
        photos=[PhotoEntry(path="a", roles=["other"])],
    )
    warnings = compute_group_warnings(g)
    assert len(warnings) == 3  # missing front, nutrition, price-tag


def test_extract_json_object_fenced() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert '"a"' in _extract_json_object(raw)


def test_extract_json_object_bare() -> None:
    assert _extract_json_object('garbage {"a": 1} more') == '{"a": 1}'


def test_extract_json_object_raises_when_missing() -> None:
    with pytest.raises(ValueError, match="No JSON object"):
        _extract_json_object("nothing here")


def test_parse_classification_happy_path() -> None:
    payload = _parse_classification(
        '{"is_same_product": true, "roles": ["front"], "rationale": "matches"}'
    )
    assert payload["is_same_product"] is True
    assert payload["roles"] == ["front"]
    assert payload["rationale"] == "matches"


def test_parse_classification_rejects_bad_is_same_product() -> None:
    with pytest.raises(ValueError, match="is_same_product"):
        _parse_classification(
            '{"is_same_product": "yes", "roles": ["front"], "rationale": ""}'
        )


def test_parse_classification_rejects_bad_role() -> None:
    with pytest.raises(ValueError, match="roles"):
        _parse_classification(
            '{"is_same_product": false, "roles": ["bogus"], "rationale": ""}'
        )


def test_parse_classification_rejects_empty_roles() -> None:
    with pytest.raises(ValueError, match="roles"):
        _parse_classification(
            '{"is_same_product": false, "roles": [], "rationale": ""}'
        )


def test_valid_roles_includes_expected_set() -> None:
    assert "front" in VALID_ROLES
    assert "nutrition" in VALID_ROLES
    assert "price-tag" in VALID_ROLES


# ---------- list_photos -------------------------------------------------------


def test_list_photos_sorted(tmp_path: Path) -> None:
    for name in ("PXL_20260426_002.jpg", "PXL_20260426_001.jpg"):
        _solid_image(tmp_path / name)
    found = [p.name for p in list_photos(tmp_path)]
    assert found == ["PXL_20260426_001.jpg", "PXL_20260426_002.jpg"]


# ---------- classify_photo (mocked) ------------------------------------------


def test_classify_photo_passes_full_context_to_claude(tmp_path: Path) -> None:
    prev_path = tmp_path / "prev.jpg"
    cur_path = tmp_path / "cur.jpg"
    front_path = tmp_path / "front.jpg"
    for p in (prev_path, cur_path, front_path):
        _solid_image(p)

    fake_response = MagicMock()
    fake_response.text = (
        '{"is_same_product": true, "roles": ["nutrition"], "rationale": "..."}'
    )
    with patch("group_photos.GROUPING_RESIZE_DIR", tmp_path / "rs"):
        with patch(
            "group_photos.call", return_value=fake_response
        ) as mock_call:
            payload = classify_photo(
                current=cur_path, previous=prev_path, front=front_path
            )

    assert payload["is_same_product"] is True
    request = mock_call.call_args.args[0]
    # Three images (previous, current, front).
    assert len(request.image_paths) == 3
    assert request.system_prompt == GROUPING_SYSTEM_PROMPT
    assert request.json_schema == GROUPING_JSON_SCHEMA


def test_classify_photo_omits_front_when_same_as_previous(
    tmp_path: Path,
) -> None:
    """If the running front IS the previous photo, don't send it twice."""
    prev_path = tmp_path / "p.jpg"
    cur_path = tmp_path / "c.jpg"
    for p in (prev_path, cur_path):
        _solid_image(p)

    fake_response = MagicMock()
    fake_response.text = (
        '{"is_same_product": false, "roles": ["front"], "rationale": "..."}'
    )
    with patch("group_photos.GROUPING_RESIZE_DIR", tmp_path / "rs"):
        with patch(
            "group_photos.call", return_value=fake_response
        ) as mock_call:
            classify_photo(
                current=cur_path, previous=prev_path, front=prev_path
            )

    request = mock_call.call_args.args[0]
    assert len(request.image_paths) == 2  # previous + current only


def test_classify_photo_handles_missing_previous_and_front(
    tmp_path: Path,
) -> None:
    cur_path = tmp_path / "c.jpg"
    _solid_image(cur_path)

    fake_response = MagicMock()
    fake_response.text = (
        '{"is_same_product": false, "roles": ["front"], "rationale": "..."}'
    )
    with patch("group_photos.GROUPING_RESIZE_DIR", tmp_path / "rs"):
        with patch(
            "group_photos.call", return_value=fake_response
        ) as mock_call:
            classify_photo(current=cur_path, previous=None, front=None)

    request = mock_call.call_args.args[0]
    assert len(request.image_paths) == 1


# ---------- assign_groups: state-machine semantics ---------------------------


def _photos_in(tmp_path: Path, names: list[str]) -> list[Path]:
    paths = []
    for name in names:
        p = tmp_path / name
        _solid_image(p)
        paths.append(p)
    return paths


def test_assign_groups_first_photo_starts_a_group_without_calling_claude(
    tmp_path: Path,
) -> None:
    paths = _photos_in(tmp_path, ["PXL_20260426_165737642.jpg"])

    with patch("group_photos.classify_photo") as mock_classify:
        groups = assign_groups(paths)

    mock_classify.assert_not_called()
    assert len(groups) == 1
    assert groups[0].store == "MOM"
    assert groups[0].photos[0].roles == ["front"]


def test_assign_groups_continues_running_group_when_same_product(
    tmp_path: Path,
) -> None:
    paths = _photos_in(
        tmp_path,
        [
            "PXL_20260426_165737642.jpg",  # MOM, first photo
            "PXL_20260426_165809667.jpg",  # MOM, same product
            "PXL_20260426_165824836.jpg",  # MOM, same product
        ],
    )

    def fake_classify(**_kwargs: object) -> dict[str, object]:
        return {
            "is_same_product": True,
            "roles": ["nutrition"],
            "rationale": "wraps around",
        }

    with patch("group_photos.classify_photo", side_effect=fake_classify):
        groups = assign_groups(paths)

    assert len(groups) == 1
    assert len(groups[0].photos) == 3


def test_assign_groups_starts_new_group_when_new_product(
    tmp_path: Path,
) -> None:
    paths = _photos_in(
        tmp_path,
        [
            "PXL_20260426_165737642.jpg",
            "PXL_20260426_165855659.jpg",  # new product
        ],
    )

    def fake_classify(**_kwargs: object) -> dict[str, object]:
        return {
            "is_same_product": False,
            "roles": ["front"],
            "rationale": "different bottle",
        }

    with patch("group_photos.classify_photo", side_effect=fake_classify):
        groups = assign_groups(paths)

    assert len(groups) == 2
    assert groups[0].id == "20260426_mom_001"
    assert groups[1].id == "20260426_mom_002"


def test_assign_groups_starts_new_group_when_store_changes(
    tmp_path: Path,
) -> None:
    """Even if the LLM says "same product", crossing the MOM/CVS cutoff splits."""
    paths = _photos_in(
        tmp_path,
        [
            "PXL_20260426_165737642.jpg",  # MOM
            CVS_CUTOFF_FILENAME,  # CVS — store boundary
        ],
    )

    def fake_classify(**_kwargs: object) -> dict[str, object]:
        return {
            "is_same_product": True,
            "roles": ["front"],
            "rationale": "wrong, but the cutoff overrides",
        }

    with patch("group_photos.classify_photo", side_effect=fake_classify):
        groups = assign_groups(paths)

    assert len(groups) == 2
    assert groups[0].store == "MOM"
    assert groups[1].store == "CVS"
    assert groups[1].id == "20260426_cvs_001"


def test_assign_groups_populates_warnings(tmp_path: Path) -> None:
    paths = _photos_in(
        tmp_path,
        [
            "PXL_20260426_165737642.jpg",  # first → "front" assumed
        ],
    )
    with patch("group_photos.classify_photo"):
        groups = assign_groups(paths)
    # Single front photo → missing nutrition + price-tag.
    assert len(groups[0].warnings) == 2


# ---------- write_groups + main() --------------------------------------------


def test_write_groups_emits_well_formed_json(tmp_path: Path) -> None:
    g = Group(
        id="20260426_mom_001",
        store="MOM",
        photos=[PhotoEntry(path="a.jpg", roles=["front"])],
        warnings=["missing price tag"],
    )
    out = tmp_path / "out.json"
    write_groups([g], out_path=out)
    payload = json.loads(out.read_text())
    assert payload["groups"][0]["id"] == "20260426_mom_001"
    assert payload["groups"][0]["photos"][0]["roles"] == ["front"]
    assert payload["groups"][0]["warnings"] == ["missing price tag"]
    assert payload["groups"][0]["locked"] is False


def test_main_runs_end_to_end_with_mocked_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ("PXL_20260426_001.jpg", "PXL_20260426_002.jpg"):
        _solid_image(raw_dir / name)
    out_path = tmp_path / "groups.json"

    monkeypatch.setattr("group_photos.RAW_PHOTOS_DIR", raw_dir)
    monkeypatch.setattr("group_photos.GROUPING_RESIZE_DIR", tmp_path / "rs")
    monkeypatch.setattr(
        "group_photos.classify_photo",
        lambda **_kw: {
            "is_same_product": True,
            "roles": ["nutrition"],
            "rationale": "x",
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        ["group_photos.py", "--out", str(out_path), "--limit", "2"],
    )

    rc = main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "Wrote" in captured.out
    payload = json.loads(out_path.read_text())
    assert len(payload["groups"]) == 1
