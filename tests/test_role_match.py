"""Tests for scripts/role_match.py — role-scoped structural comparators."""

from __future__ import annotations

from role_extraction import (
    UNKNOWN_MARKER,
    FrontExtraction,
    NutritionTable,
    PriceTag,
)
from role_match import (
    DEFAULT_MATCH_THRESHOLD,
    matches,
    score_front,
    score_nutrition,
    score_price_tag,
)


# ---------------------------------------------------------------------------
# Front
# ---------------------------------------------------------------------------


def test_score_front_identical_returns_one() -> None:
    a = FrontExtraction(product_name="men's one", manufacturer="rainbow light")
    assert score_front(a, a) == 1.0


def test_score_front_both_empty_manufacturer_treated_as_match() -> None:
    """A panel with no manufacturer line shouldn't penalize the score —
    both runs correctly returning empty manufacturer is agreement."""
    a = FrontExtraction(product_name="men's one", manufacturer="")
    b = FrontExtraction(product_name="men's one", manufacturer="")
    assert score_front(a, b) == 1.0


def test_score_front_partial_name_mismatch_lowers_score() -> None:
    a = FrontExtraction(product_name="men's one", manufacturer="rainbow light")
    b = FrontExtraction(product_name="womens one", manufacturer="rainbow light")
    score = score_front(a, b)
    assert 0.5 < score < 1.0


def test_score_front_completely_different_returns_low_score() -> None:
    a = FrontExtraction(product_name="aaaa", manufacturer="bbbb")
    b = FrontExtraction(product_name="zzzz", manufacturer="yyyy")
    assert score_front(a, b) == 0.0


# ---------------------------------------------------------------------------
# Nutrition table
# ---------------------------------------------------------------------------


def test_score_nutrition_identical_returns_one() -> None:
    table = NutritionTable(
        rows=(
            ("calories", "6", "15"),
            ("vitamin a", "225 mcg", "75%", "450 mcg", "75%"),
        )
    )
    assert score_nutrition(table, table) == 1.0


def test_score_nutrition_both_empty_treated_as_match() -> None:
    assert score_nutrition(NutritionTable(()), NutritionTable(())) == 1.0


def test_score_nutrition_one_empty_returns_zero() -> None:
    table = NutritionTable(rows=(("calories", "6"),))
    assert score_nutrition(table, NutritionTable(())) == 0.0
    assert score_nutrition(NutritionTable(()), table) == 0.0


def test_score_nutrition_row_order_independent() -> None:
    """Rows are matched by their first cell, so row order doesn't matter —
    this is the whole point vs. text-distance comparison."""
    a = NutritionTable(
        rows=(
            ("calories", "6", "15"),
            ("protein", "0g", "0g"),
        )
    )
    b = NutritionTable(
        rows=(
            ("protein", "0g", "0g"),
            ("calories", "6", "15"),
        )
    )
    assert score_nutrition(a, b) == 1.0


def test_score_nutrition_value_mismatch_lowers_score() -> None:
    a = NutritionTable(rows=(("calories", "6", "15"), ("protein", "0g", "0g")))
    b = NutritionTable(
        rows=(
            ("calories", "6", "15"),
            ("protein", "1g", "0g"),  # one cell different
        )
    )
    score = score_nutrition(a, b)
    # 5 of 6 cells match (calories: 3/3, protein: 2/3).
    assert score == 5 / 6


def test_score_nutrition_missing_row_counts_as_full_mismatch() -> None:
    a = NutritionTable(
        rows=(("calories", "6", "15"), ("vitamin a", "225 mcg", "75%"))
    )
    b = NutritionTable(rows=(("calories", "6", "15"),))
    score = score_nutrition(a, b)
    # 3 calorie cells match, 3 vitamin-a cells in a have no counterpart in
    # b → 3 of 6 = 0.5.
    assert score == 0.5


def test_score_nutrition_unknown_marker_excluded_from_comparison() -> None:
    """An unreadable cell shouldn't be charged against the model — it's
    uncomparable and drops out of both numerator and denominator."""
    a = NutritionTable(rows=(("calories", "6", "15"),))
    b = NutritionTable(rows=(("calories", UNKNOWN_MARKER, "15"),))
    # Without the marker exclusion, this would be 2/3. With it, the
    # masked cell drops out → 2/2 = 1.0.
    assert score_nutrition(a, b) == 1.0


def test_score_nutrition_all_unknown_returns_one() -> None:
    """If every comparable cell is masked, the comparison is vacuous —
    treat as a match rather than dividing zero by zero. The nutrient-name
    column matches normally; only the value cells are masked."""
    a = NutritionTable(rows=(("calories", UNKNOWN_MARKER),))
    b = NutritionTable(rows=(("calories", UNKNOWN_MARKER),))
    # 1 of 1 comparable cells match → 1.0
    assert score_nutrition(a, b) == 1.0


def test_score_nutrition_only_masked_cells_treated_as_match() -> None:
    """Edge case: a row where every cell — including the key — is masked.
    The two tables agree that the row exists but neither can read any of
    it; with all comparable cells dropped, total_cells is 0 and we
    short-circuit to 1.0 rather than divide by zero."""
    a = NutritionTable(rows=((UNKNOWN_MARKER, UNKNOWN_MARKER),))
    b = NutritionTable(rows=((UNKNOWN_MARKER, UNKNOWN_MARKER),))
    assert score_nutrition(a, b) == 1.0


# ---------------------------------------------------------------------------
# Price tag
# ---------------------------------------------------------------------------


def _tag(
    *,
    store: str = "MOM",
    price_cents: int = 1599,
    description: str = "nc lqud mltvitmn 30 fz",
    size: str = "30 fz",
    upc: str = "727783904379",
) -> PriceTag:
    return PriceTag(
        store=store,
        price_cents=price_cents,
        description=description,
        size=size,
        upc=upc,
    )


def test_score_price_tag_identical_returns_one() -> None:
    t = _tag()
    assert score_price_tag(t, t) == 1.0


def test_score_price_tag_store_mismatch_returns_zero() -> None:
    """Cross-store comparisons are nonsensical — UPC semantics differ."""
    a = _tag(store="MOM", upc="727783904379")
    b = _tag(store="CVS", upc="01790")
    assert score_price_tag(a, b) == 0.0


def test_score_price_tag_upc_off_by_one_digit_drops_score() -> None:
    """UPC equality is binary — one digit off means wrong product."""
    a = _tag(upc="727783904379")
    b = _tag(upc="727783904378")
    score = score_price_tag(a, b)
    # 3 of 4 fields match (price, description, size) → 0.75
    assert score == 0.75


def test_score_price_tag_price_off_drops_score() -> None:
    a = _tag(price_cents=1599)
    b = _tag(price_cents=1499)
    score = score_price_tag(a, b)
    assert score == 0.75


def test_score_price_tag_both_empty_size_treated_as_match() -> None:
    a = _tag(size="")
    b = _tag(size="")
    assert score_price_tag(a, b) == 1.0


# ---------------------------------------------------------------------------
# Match threshold
# ---------------------------------------------------------------------------


def test_matches_uses_default_threshold() -> None:
    assert matches(0.95) is True
    assert matches(DEFAULT_MATCH_THRESHOLD) is True
    assert matches(DEFAULT_MATCH_THRESHOLD - 0.01) is False


def test_matches_accepts_custom_threshold() -> None:
    assert matches(0.5, threshold=0.4) is True
    assert matches(0.5, threshold=0.6) is False
