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
    """Cell-by-cell match on rows keyed by their first cell (nutrient name).

    * Two empty tables → ``1.0``. One empty → ``0.0``.
    * Rows are matched by their first cell (typically the nutrient name).
      Rows whose first cell exists in only one of the two tables count
      every cell as a mismatch in the denominator.
    * :data:`role_extraction.UNKNOWN_MARKER` cells are uncomparable and
      excluded from both numerator and denominator — an unreadable cell
      shouldn't be charged against the model.
    """
    if not a.rows and not b.rows:
        return 1.0
    if not a.rows or not b.rows:
        return 0.0

    rows_a = {row[0]: row for row in a.rows if row}
    rows_b = {row[0]: row for row in b.rows if row}
    all_keys = set(rows_a) | set(rows_b)

    matched_cells = 0
    total_cells = 0
    for key in all_keys:
        row_a = rows_a.get(key, ())
        row_b = rows_b.get(key, ())
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
        # Every comparable cell was masked by UNKNOWN_MARKER; nothing to
        # compare. Treat as a match — the model didn't claim any value
        # we could disagree with.
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
