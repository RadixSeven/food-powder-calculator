"""Tests for scripts/extract_yaml.py — per-group YAML extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from PIL import Image

from extract_yaml import (
    GroupExtraction,
    _extract_nutrition_with_front_context,
    extract_group,
    process_group_to_yaml,
    write_extraction_yaml,
)
from group_pipeline import GroupArtifacts
from role_extraction import FrontExtraction, NutritionTable, PriceTag


def _save(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    Image.new("RGB", size, (200, 100, 50)).save(path, "JPEG", quality=85)


# ---------------------------------------------------------------------------
# extract_group — dispatches per role
# ---------------------------------------------------------------------------


def test_extract_group_calls_each_role_extractor(tmp_path: Path) -> None:
    front_img = tmp_path / "front.jpg"
    nutrition_img = tmp_path / "nutrition.jpg"
    tag_img = tmp_path / "tag.jpg"
    for p in (front_img, nutrition_img, tag_img):
        _save(p)

    artifacts = GroupArtifacts(
        group_id="g1",
        store="MOM",
        panels_by_role={
            "front": front_img,
            "nutrition": nutrition_img,
            "price-tag": tag_img,
        },
        stitched={"front": False, "nutrition": False, "price-tag": False},
    )

    front_result = FrontExtraction(product_name="x", manufacturer="y")
    nutrition_result = NutritionTable(rows=(("calories", "100"),))
    tag_result = PriceTag(
        store="MOM",
        price_cents=199,
        description="x",
        size="",
        upc="123456789012",
    )

    with patch("extract_yaml.extract_front", return_value=front_result):
        with patch(
            "extract_yaml._extract_nutrition_with_front_context",
            return_value=nutrition_result,
        ) as mock_nutr:
            with patch(
                "extract_yaml.extract_price_tag", return_value=tag_result
            ) as mock_tag:
                result = extract_group(artifacts)

    # Front context is forwarded to nutrition extraction.
    assert mock_nutr.call_args.args == (nutrition_img, front_result)
    # Store is forwarded to price-tag extraction (MOM vs CVS prompt).
    assert mock_tag.call_args.kwargs["store"] == "MOM"
    assert result.front == front_result
    assert result.nutrition == nutrition_result
    assert result.price_tag == tag_result


def test_extract_group_skips_missing_roles(tmp_path: Path) -> None:
    """A group missing nutrition (no nutrition shot in gold) leaves the
    nutrition field None rather than calling the extractor."""
    front_img = tmp_path / "front.jpg"
    _save(front_img)
    artifacts = GroupArtifacts(
        group_id="g1",
        store="MOM",
        panels_by_role={"front": front_img},
        stitched={"front": False},
    )
    with patch(
        "extract_yaml.extract_front",
        return_value=FrontExtraction(product_name="x", manufacturer=""),
    ):
        with patch(
            "extract_yaml._extract_nutrition_with_front_context"
        ) as mock_nutr:
            with patch("extract_yaml.extract_price_tag") as mock_tag:
                result = extract_group(artifacts)
    mock_nutr.assert_not_called()
    mock_tag.assert_not_called()
    assert result.nutrition is None
    assert result.price_tag is None


# ---------------------------------------------------------------------------
# Front context augmentation
# ---------------------------------------------------------------------------


def test_nutrition_with_front_context_augments_system_prompt(
    tmp_path: Path,
) -> None:
    """Front product name is prepended to the nutrition system prompt
    so the model can disambiguate multi-column %DV tables."""
    img = tmp_path / "n.jpg"
    _save(img)
    front = FrontExtraction(
        product_name="children's liquid multi", manufacturer="new chapter"
    )
    from _claude import ClaudeResponse

    fake = ClaudeResponse(
        text='{"rows": [["calories", "10"]]}', cached=False, request_sha="x"
    )
    with patch("extract_yaml.call", return_value=fake) as mock_call:
        result = _extract_nutrition_with_front_context(img, front)
    request = mock_call.call_args.args[0]
    sp = request.system_prompt or ""
    assert "children's liquid multi" in sp
    assert "new chapter" in sp
    assert result.rows == (("calories", "10"),)


def test_nutrition_without_front_falls_back_to_bare_prompt(
    tmp_path: Path,
) -> None:
    """If front extraction failed (front=None), use the unmodified
    nutrition extractor."""
    img = tmp_path / "n.jpg"
    _save(img)
    fake_table = NutritionTable(rows=(("x", "y"),))
    with patch(
        "extract_yaml.extract_nutrition", return_value=fake_table
    ) as mock_nutr:
        with patch("extract_yaml.call") as mock_call:
            result = _extract_nutrition_with_front_context(img, None)
    mock_nutr.assert_called_once()
    mock_call.assert_not_called()
    assert result == fake_table


def test_nutrition_with_front_omits_manufacturer_when_empty(
    tmp_path: Path,
) -> None:
    """An empty manufacturer field shouldn't pollute the context line
    with `by ''`. The augmentation drops the `by X` clause entirely."""
    img = tmp_path / "n.jpg"
    _save(img)
    front = FrontExtraction(product_name="some product", manufacturer="")
    from _claude import ClaudeResponse

    fake = ClaudeResponse(text='{"rows": []}', cached=False, request_sha="x")
    with patch("extract_yaml.call", return_value=fake) as mock_call:
        _extract_nutrition_with_front_context(img, front)
    sp = mock_call.call_args.args[0].system_prompt or ""
    assert "some product" in sp
    assert " by " not in sp


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------


def test_write_extraction_yaml_includes_all_present_fields(
    tmp_path: Path,
) -> None:
    extraction = GroupExtraction(
        group_id="g1",
        store="MOM",
        front=FrontExtraction(product_name="p", manufacturer="m"),
        nutrition=NutritionTable(rows=(("a", "b"), ("c", "d"))),
        price_tag=PriceTag(
            store="MOM",
            price_cents=1999,
            description="p",
            size="30 fz",
            upc="123456789012",
        ),
    )
    out = tmp_path / "g1.yaml"
    write_extraction_yaml(extraction, out)
    parsed = yaml.safe_load(out.read_text())
    assert parsed["group_id"] == "g1"
    assert parsed["store"] == "MOM"
    assert parsed["front"]["product_name"] == "p"
    assert parsed["nutrition"]["rows"] == [["a", "b"], ["c", "d"]]
    assert parsed["price_tag"]["price_cents"] == 1999


def test_write_extraction_yaml_skips_none_fields(tmp_path: Path) -> None:
    """When a role wasn't extracted (gold didn't have it), the YAML
    omits the key rather than writing 'null' — keeps human review clean."""
    extraction = GroupExtraction(
        group_id="g1",
        store="MOM",
        front=FrontExtraction(product_name="p", manufacturer=""),
        nutrition=None,
        price_tag=None,
    )
    out = tmp_path / "g1.yaml"
    write_extraction_yaml(extraction, out)
    parsed = yaml.safe_load(out.read_text())
    assert "nutrition" not in parsed
    assert "price_tag" not in parsed


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def test_process_group_to_yaml_writes_yaml_at_expected_path(
    tmp_path: Path,
) -> None:
    """Smoke test that the full path (load gold → bbox → crop → stitch →
    extract → write) wires up correctly when the LLM-bound parts are
    mocked. Validates the integration glue, not the extraction quality."""
    gold = tmp_path / "gold.json"
    img = tmp_path / "img.jpg"
    _save(img)
    gold.write_text(
        f'{{"groups": [{{"id": "g1", "photos": ['
        f'{{"path": "{img.name}", "roles": ["front"]}}]}}]}}'
    )

    fake_artifacts = GroupArtifacts(
        group_id="g1",
        store="MOM",
        panels_by_role={"front": img},
        stitched={"front": False},
    )
    fake_extraction = GroupExtraction(
        group_id="g1",
        store="MOM",
        front=FrontExtraction(product_name="x", manufacturer=""),
        nutrition=None,
        price_tag=None,
    )
    with patch("extract_yaml.detect_and_crop_group", return_value={}):
        with patch(
            "extract_yaml.stitch_role_outputs", return_value=fake_artifacts
        ):
            with patch(
                "extract_yaml.extract_group", return_value=fake_extraction
            ):
                out = process_group_to_yaml(
                    "g1",
                    gold_path=gold,
                    stitched_root=tmp_path / "stitched",
                    out_dir=tmp_path / "yaml",
                )
    assert out == tmp_path / "yaml" / "g1.yaml"
    parsed = yaml.safe_load(out.read_text())
    assert parsed["group_id"] == "g1"


def test_process_group_to_yaml_propagates_missing_group_id(
    tmp_path: Path,
) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text('{"groups": []}')
    with pytest.raises(KeyError):
        process_group_to_yaml(
            "nonexistent",
            gold_path=gold,
            stitched_root=tmp_path / "stitched",
            out_dir=tmp_path / "yaml",
        )


def test_process_group_to_yaml_skips_existing_yaml_by_default(
    tmp_path: Path,
) -> None:
    """A pre-existing YAML for a group short-circuits the whole pipeline
    without calling the extractor — so resuming a long batch doesn't
    pay for groups already processed."""
    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()
    pre_existing = yaml_dir / "g1.yaml"
    pre_existing.write_text("group_id: g1\nstore: MOM\n")

    with patch("extract_yaml.load_group") as mock_load:
        with patch("extract_yaml.detect_and_crop_group") as mock_detect:
            with patch("extract_yaml.extract_group") as mock_extract:
                out = process_group_to_yaml(
                    "g1",
                    gold_path=tmp_path / "gold.json",
                    stitched_root=tmp_path / "stitched",
                    out_dir=yaml_dir,
                )
    mock_load.assert_not_called()
    mock_detect.assert_not_called()
    mock_extract.assert_not_called()
    assert out == pre_existing
    # Original file content should be preserved (not rewritten).
    assert pre_existing.read_text() == "group_id: g1\nstore: MOM\n"


def test_process_group_to_yaml_force_reextracts(tmp_path: Path) -> None:
    """skip_if_exists=False rebuilds the YAML even if one exists —
    the path used by the --force CLI flag for re-running after a
    config change."""
    gold = tmp_path / "gold.json"
    img = tmp_path / "p.jpg"
    _save(img)
    gold.write_text(
        f'{{"groups": [{{"id": "g1", "photos": ['
        f'{{"path": "{img.name}", "roles": ["front"]}}]}}]}}'
    )
    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()
    (yaml_dir / "g1.yaml").write_text("stale: true\n")

    fake_artifacts = GroupArtifacts(
        group_id="g1",
        store="MOM",
        panels_by_role={"front": img},
        stitched={"front": False},
    )
    fake_extraction = GroupExtraction(
        group_id="g1",
        store="MOM",
        front=FrontExtraction(product_name="x", manufacturer=""),
        nutrition=None,
        price_tag=None,
    )
    with patch("extract_yaml.detect_and_crop_group", return_value={}):
        with patch(
            "extract_yaml.stitch_role_outputs", return_value=fake_artifacts
        ):
            with patch(
                "extract_yaml.extract_group", return_value=fake_extraction
            ):
                process_group_to_yaml(
                    "g1",
                    gold_path=gold,
                    stitched_root=tmp_path / "stitched",
                    out_dir=yaml_dir,
                    skip_if_exists=False,
                )
    # YAML overwritten with fresh extraction, not preserved as 'stale: true'.
    parsed = yaml.safe_load((yaml_dir / "g1.yaml").read_text())
    assert parsed["group_id"] == "g1"
