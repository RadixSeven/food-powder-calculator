"""Structural comparison of role-specific extractions.

Replaces the global Levenshtein-against-opus metric used in
:mod:`find_legible_size` with a per-role, per-field comparator that
ignores irrelevant background and is robust to reading-order
divergence. Used by the binary-search sizing finder to decide whether a
smaller-model probe at size *X* is "close enough" to the opus reference
on the relevant role.

Each scoring function returns a value in ``[0.0, 1.0]``:

* ``1.0`` — identical content.
* ``0.0`` — no overlap (different stores, no matching cells, etc.).

The :data:`DEFAULT_MATCH_THRESHOLD` is what the binary search uses for
the boolean "matched / didn't match" decision; it can be lowered for
very dense panels where even two opus runs disagree slightly.

Why structural rather than text-distance:

* The nutrition table is the only role with row/column structure to
  compare. Cell-by-cell matching ignores how the model serialized rows
  and naturally tolerates partial extractions (one model misses two
  nutrients but reads the rest correctly).
* Front and price-tag are flat field sets; per-field scoring lets a
  single bad field (e.g., one digit off in the UPC) influence the
  overall score in a calibrated way rather than getting drowned out by
  matching surrounding text.
* No LLM-as-judge: deterministic, fast, no extra calls in the sizing
  inner loop.
"""

from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher

from role_extraction import (
    UNKNOWN_MARKER,
    FrontExtraction,
    NutritionTable,
    PriceTag,
)

# Threshold above which a sizing probe is considered to have matched the
# reference. Tuned conservatively: 0.90 leaves headroom for the small
# stochasticity that's present even in two opus runs of the same image.
DEFAULT_MATCH_THRESHOLD = 0.90


def score_front(a: FrontExtraction, b: FrontExtraction) -> float:
    """Average per-field Levenshtein ratio across product_name + manufacturer.

    Both fields are weighted equally. An empty manufacturer on both sides
    contributes a perfect 1.0 (the panel simply didn't print one).
    """
    name_ratio = SequenceMatcher(None, a.product_name, b.product_name).ratio()
    mfr_ratio = _string_ratio(a.manufacturer, b.manufacturer)
    return (name_ratio + mfr_ratio) / 2


def score_nutrition(a: NutritionTable, b: NutritionTable) -> float:
    """Cell-by-cell match across two nutrition tables.

    Two strategies, picked per-call:

    * **Name-keyed** (when both tables have usable nutrient names in
      cell 0): rows in A and B are matched by their cell-0 string.
      Robust to row-reordering between extractions; the right choice
      when the full panel is in frame.
    * **Position-keyed** (fallback when either table lacks usable
      names): row N in A matches row N in B. Necessary for partial
      wrap-around views where the nutrient-name column is off-frame
      (e.g., the right half of a bottle wrap-around shows only value /
      %DV columns; cell 0 is then the first value column, and many
      rows share the same first value — name-keyed would collide).

    In both modes :data:`role_extraction.UNKNOWN_MARKER` cells are
    uncomparable and excluded from both numerator and denominator.

    Two empty tables → ``1.0``; one empty → ``0.0``.
    """
    if not a.rows and not b.rows:
        return 1.0
    if not a.rows or not b.rows:
        return 0.0

    if _has_usable_names(a) and _has_usable_names(b):
        return _score_nutrition_by_name(a, b)
    return _score_nutrition_by_position(a, b)


def _has_usable_names(table: NutritionTable) -> bool:
    """Return True iff at least half the rows have a distinct, readable name.

    The "distinct" check rules out tables where cell 0 is actually a value
    column (many duplicates, e.g. ``0.63 mg`` repeated across multiple
    B-vitamin rows).

    Caller (:func:`score_nutrition`) handles the empty-table case
    before calling, so we don't repeat the guard here.
    """
    seen: set[str] = set()
    for row in table.rows:
        if row and row[0] and row[0] != UNKNOWN_MARKER:
            seen.add(row[0])
    return len(seen) >= max(1, len(table.rows) // 2)


def _score_nutrition_by_name(a: NutritionTable, b: NutritionTable) -> float:
    rows_a = {row[0]: row for row in a.rows if row}
    rows_b = {row[0]: row for row in b.rows if row}
    return _score_row_pairs(
        (
            (rows_a.get(k, ()), rows_b.get(k, ()))
            for k in set(rows_a) | set(rows_b)
        )
    )


def _score_nutrition_by_position(a: NutritionTable, b: NutritionTable) -> float:
    n = max(len(a.rows), len(b.rows))

    def pair(i: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (
            a.rows[i] if i < len(a.rows) else (),
            b.rows[i] if i < len(b.rows) else (),
        )

    return _score_row_pairs(pair(i) for i in range(n))


def _score_row_pairs(
    pairs: Iterable[tuple[tuple[str, ...], tuple[str, ...]]],
) -> float:
    matched_cells = 0
    total_cells = 0
    for row_a, row_b in pairs:
        n = max(len(row_a), len(row_b))
        for i in range(n):
            ca = row_a[i] if i < len(row_a) else ""
            cb = row_b[i] if i < len(row_b) else ""
            if ca == UNKNOWN_MARKER or cb == UNKNOWN_MARKER:
                continue
            total_cells += 1
            if ca == cb:
                matched_cells += 1
    if total_cells == 0:
        # Every comparable cell was masked by UNKNOWN_MARKER, or both
        # tables happened to be all-empty rows; nothing to compare.
        return 1.0
    return matched_cells / total_cells


def score_price_tag(a: PriceTag, b: PriceTag) -> float:
    """Per-field score with binary equality on price/UPC and string ratio
    on description/size.

    A store mismatch returns 0.0 immediately — the comparison is
    nonsensical across stores (different UPC semantics).
    """
    if a.store != b.store:
        return 0.0
    fields: list[float] = [
        1.0 if a.price_cents == b.price_cents else 0.0,
        SequenceMatcher(None, a.description, b.description).ratio(),
        _string_ratio(a.size, b.size),
        1.0 if a.upc == b.upc else 0.0,
    ]
    return sum(fields) / len(fields)


def matches(score: float, threshold: float = DEFAULT_MATCH_THRESHOLD) -> bool:
    """Boolean wrapper for binary-search probes."""
    return score >= threshold


def _string_ratio(a: str, b: str) -> float:
    """Sequence ratio with the both-empty corner case treated as match.

    ``SequenceMatcher`` returns 0.0 for two empty strings, which is
    counter-intuitive: if the panel doesn't print a manufacturer line,
    both extractions correctly say "" and that should count as agreement.
    """
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()
