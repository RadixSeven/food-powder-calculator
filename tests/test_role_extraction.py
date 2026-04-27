"""Tests for scripts/role_extraction.py — role-specific extraction prompts.

The `_claude.call` subprocess wrapper is mocked so these tests never hit
the network. We assert on the parsed dataclass shape and on the prompt
selection (MOM vs CVS) so a future prompt rewording doesn't silently
swap which store gets the UPC vs the slug.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from _claude import ClaudeResponse
from role_extraction import (
    PRICE_TAG_CVS_SYSTEM_PROMPT,
    PRICE_TAG_MOM_SYSTEM_PROMPT,
    UNKNOWN_MARKER,
    FrontExtraction,
    NutritionTable,
    PriceTag,
    extract_front,
    extract_nutrition,
    extract_price_tag,
)


def _stub_response(payload: object) -> ClaudeResponse:
    return ClaudeResponse(
        text=json.dumps(payload), cached=False, request_sha="x"
    )


# ---------------------------------------------------------------------------
# Front
# ---------------------------------------------------------------------------


def test_extract_front_parses_required_fields(tmp_path: Path) -> None:
    img = tmp_path / "front.jpg"
    img.write_bytes(b"fake")
    fake = _stub_response(
        {
            "product_name": "rainbow light men's one",
            "manufacturer": "rainbow light",
        }
    )
    with patch("role_extraction.call", return_value=fake):
        result = extract_front(img, "haiku")
    assert result == FrontExtraction(
        product_name="rainbow light men's one",
        manufacturer="rainbow light",
    )


def test_extract_front_allows_empty_manufacturer(tmp_path: Path) -> None:
    """Some front panels print a marketing name with no manufacturer line."""
    img = tmp_path / "front.jpg"
    img.write_bytes(b"fake")
    fake = _stub_response({"product_name": "men's one", "manufacturer": ""})
    with patch("role_extraction.call", return_value=fake):
        result = extract_front(img, "haiku")
    assert result.manufacturer == ""


# ---------------------------------------------------------------------------
# Nutrition table
# ---------------------------------------------------------------------------


def test_extract_nutrition_returns_immutable_nested_tuples(
    tmp_path: Path,
) -> None:
    """Nested arrays come back as a frozen dataclass of tuples-of-tuples
    so downstream code can use it as a hash / dict key without copying."""
    img = tmp_path / "panel.jpg"
    img.write_bytes(b"fake")
    fake = _stub_response(
        {
            "rows": [
                ["amount per serving", "2-3 years", "4-13 years"],
                ["calories", "6", "15"],
                ["vitamin a", "225 mcg", "75%", "450 mcg", "75%"],
            ]
        }
    )
    with patch("role_extraction.call", return_value=fake):
        table = extract_nutrition(img, "opus")
    assert isinstance(table, NutritionTable)
    assert table.rows == (
        ("amount per serving", "2-3 years", "4-13 years"),
        ("calories", "6", "15"),
        ("vitamin a", "225 mcg", "75%", "450 mcg", "75%"),
    )


def test_extract_nutrition_preserves_unknown_marker(tmp_path: Path) -> None:
    """The model emits the special marker for unreadable cells; the parser
    must round-trip it unchanged so the comparator can treat it specially."""
    img = tmp_path / "panel.jpg"
    img.write_bytes(b"fake")
    fake = _stub_response(
        {"rows": [["calories", UNKNOWN_MARKER, "15"], ["protein", "", "0g"]]}
    )
    with patch("role_extraction.call", return_value=fake):
        table = extract_nutrition(img, "opus")
    assert table.rows[0][1] == UNKNOWN_MARKER
    assert table.rows[1][1] == ""  # blank cell preserved as empty string


def test_extract_nutrition_handles_empty_table(tmp_path: Path) -> None:
    """A photo with no readable nutrition table returns rows=()."""
    img = tmp_path / "panel.jpg"
    img.write_bytes(b"fake")
    fake = _stub_response({"rows": []})
    with patch("role_extraction.call", return_value=fake):
        table = extract_nutrition(img, "opus")
    assert table.rows == ()


# ---------------------------------------------------------------------------
# Price tag — store-conditional prompt selection
# ---------------------------------------------------------------------------


def test_extract_price_tag_mom_uses_mom_prompt(tmp_path: Path) -> None:
    img = tmp_path / "tag.jpg"
    img.write_bytes(b"fake")
    fake = _stub_response(
        {
            "price_cents": 1599,
            "description": "nc lqud mltvitmn orng mng 30 fz",
            "size": "30 fz",
            "upc": "727783904379",
        }
    )
    with patch("role_extraction.call", return_value=fake) as mock_call:
        result = extract_price_tag(img, "haiku", store="MOM")
    request = mock_call.call_args.args[0]
    assert request.system_prompt == PRICE_TAG_MOM_SYSTEM_PROMPT
    assert result == PriceTag(
        store="MOM",
        price_cents=1599,
        description="nc lqud mltvitmn orng mng 30 fz",
        size="30 fz",
        upc="727783904379",
    )


def test_extract_price_tag_cvs_uses_cvs_prompt(tmp_path: Path) -> None:
    img = tmp_path / "tag.jpg"
    img.write_bytes(b"fake")
    fake = _stub_response(
        {
            "price_cents": 1279,
            "description": "nm mv him 50+",
            "size": "90ct",
            "upc": "01790",  # the 5-digit I3* slug, not a 12-digit UPC
        }
    )
    with patch("role_extraction.call", return_value=fake) as mock_call:
        result = extract_price_tag(img, "haiku", store="CVS")
    request = mock_call.call_args.args[0]
    assert request.system_prompt == PRICE_TAG_CVS_SYSTEM_PROMPT
    assert result.store == "CVS"
    assert result.upc == "01790"
    assert len(result.upc) == 5


def test_extract_price_tag_rejects_unknown_store(tmp_path: Path) -> None:
    img = tmp_path / "tag.jpg"
    img.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Unknown store"):
        extract_price_tag(img, "haiku", store="WHOLEFOODS")


def test_extract_price_tag_allows_empty_upc_when_unreadable(
    tmp_path: Path,
) -> None:
    """If the model can't read the UPC slug, it returns "" — the comparator
    handles a missing UPC as a low-confidence cross-check rather than a
    confident mismatch."""
    img = tmp_path / "tag.jpg"
    img.write_bytes(b"fake")
    fake = _stub_response(
        {
            "price_cents": 1499,
            "description": "some product",
            "size": "",
            "upc": "",
        }
    )
    with patch("role_extraction.call", return_value=fake):
        result = extract_price_tag(img, "haiku", store="MOM")
    assert result.upc == ""
    assert result.size == ""
