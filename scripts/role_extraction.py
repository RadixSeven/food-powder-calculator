"""Role-specific structured extraction from product photos.

Three extraction tasks, one per role, each with its own prompt, JSON
schema, and parsed dataclass:

- :func:`extract_front` — product name + manufacturer (front-of-package).
- :func:`extract_nutrition` — the supplement-facts / nutrition-facts
  table as a nested array of strings (rows × cells).
- :func:`extract_price_tag` — store-conditional price-tag fields. For
  MOM the full 12-digit UPC; for CVS the 5-digit "I3* DDDDD" slug from
  the tag header.

The same prompts drive both final extraction (:mod:`extract` later) and
the role-scoped sizing comparator (:mod:`role_match`). Generic-text
extraction had two failure modes that role-scoped extraction avoids:

1. Sizing probes diverge mostly on irrelevant background (other shelf
   bottles, hands, bottle bottoms). Asking only for the role-relevant
   payload removes that noise.
2. Multi-column tables (kid-vs-adult %DV) get reading-order ambiguity
   that bloats Levenshtein distance even when the underlying values
   match. The nested-array format pins the column order explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from _claude import ClaudeRequest, call

# Marker the model emits for cells whose text it cannot read confidently.
# Renders as a small filled rectangle in any monospace font and is unlikely
# to occur as real on-package text.
UNKNOWN_MARKER = "▮"  # ▮ BLACK VERTICAL RECTANGLE


# ---------------------------------------------------------------------------
# Front-of-package: product name + manufacturer
# ---------------------------------------------------------------------------

FRONT_SYSTEM_PROMPT = (
    "You extract a small set of fields from a front-of-package product "
    "photo for a deterministic comparison task. Return a JSON object with "
    "two keys: "
    '`"product_name"` (the full product name as it appears on the front '
    "panel, including flavor / variant / count if printed there, "
    "lower-cased, single-spaced); and "
    '`"manufacturer"` (the manufacturer / brand owner exactly as printed, '
    "lower-cased; empty string if no manufacturer name is visible). "
    "Return JSON only, no prose."
)

FRONT_PROMPT = (
    "Extract the product name and manufacturer from this front-of-package "
    "photo, following the instructions in the system prompt."
)

FRONT_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "product_name": {"type": "string"},
            "manufacturer": {"type": "string"},
        },
        "required": ["product_name", "manufacturer"],
        "additionalProperties": False,
    }
)


@dataclass(frozen=True)
class FrontExtraction:
    """Front-of-package fields used to anchor downstream prompts."""

    product_name: str
    manufacturer: str  # empty string if not visible


def extract_front(image_path: Path, model: str) -> FrontExtraction:
    """Pull product name + manufacturer from a front-of-package photo."""
    response = call(
        ClaudeRequest(
            prompt=FRONT_PROMPT,
            model=model,
            image_paths=(image_path,),
            system_prompt=FRONT_SYSTEM_PROMPT,
            json_schema=FRONT_JSON_SCHEMA,
        )
    )
    payload = json.loads(response.text)
    return FrontExtraction(
        product_name=str(payload["product_name"]),
        manufacturer=str(payload["manufacturer"]),
    )


# ---------------------------------------------------------------------------
# Nutrition table: nested rows of strings
# ---------------------------------------------------------------------------

NUTRITION_SYSTEM_PROMPT = (
    "You extract the nutrition / supplement facts table from a product "
    "photo for a deterministic comparison task. Return a JSON object with "
    'one key, `"rows"`, whose value is an array of arrays of strings. '
    "Each inner array is one row of the table, in top-to-bottom order. "
    "Within a row, cells go left-to-right in the column order printed on "
    "the panel; do not reorder or merge columns. "
    "Header rows (e.g. `Amount per serving`, age-band columns) are rows "
    "too — emit them in source order so column meaning is preserved. "
    'Use an empty string `""` for cells the panel leaves blank. Use '
    f'`"{UNKNOWN_MARKER}"` for cells whose text is occluded, glared, '
    "cut off, or otherwise unreadable. Lower-case all text; preserve "
    "punctuation (parentheses, percent signs, decimal points). Do not "
    "invent values or fill in nutrients that are not visible. "
    "Return JSON only, no prose."
)

NUTRITION_PROMPT = (
    "Extract the nutrition / supplement facts table from this photo, "
    "following the instructions in the system prompt."
)

NUTRITION_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string"}},
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    }
)


@dataclass(frozen=True)
class NutritionTable:
    """A nutrition / supplement facts table as rows of cells."""

    rows: tuple[tuple[str, ...], ...]


NUTRITION_MULTI_SYSTEM_PROMPT = (
    "You extract the nutrition / supplement facts table from photos "
    "of a product. You may receive multiple photos covering different "
    "portions of the same panel — e.g. a wrap-around bottle label "
    "shot from different angles, or partial views where each photo "
    "shows a slice of the table. Combine the rows you can see across "
    "all photos into a single table, deduplicating rows that appear "
    "in more than one photo (matching by nutrient name when "
    "available, otherwise by row position within each photo). "
    'Return a JSON object with one key, `"rows"`, whose value is an '
    "array of arrays of strings. Each inner array is one row of the "
    "combined table, in top-to-bottom order as they appear on the "
    "panel. Within a row, cells go left-to-right in the column order "
    "printed on the panel; do not reorder or merge columns. Header "
    "rows (e.g. `Amount per serving`, age-band columns) are rows too "
    "— emit them in source order so column meaning is preserved. Use "
    'an empty string `""` for cells the panel leaves blank. Use '
    f'`"{UNKNOWN_MARKER}"` only for cells whose text is genuinely '
    "occluded across all photos (no photo shows that cell readable). "
    "If a cell is readable in any photo, prefer that reading. "
    "Lower-case all text; preserve punctuation. Do not invent values. "
    "Return JSON only, no prose."
)

NUTRITION_MULTI_PROMPT = (
    "Extract the combined nutrition / supplement facts table from "
    "these {n} photos of the same product, following the "
    "instructions in the system prompt."
)


def extract_nutrition(image_path: Path, model: str) -> NutritionTable:
    """Extract the nutrition table from a single (typically cropped) panel photo."""
    response = call(
        ClaudeRequest(
            prompt=NUTRITION_PROMPT,
            model=model,
            image_paths=(image_path,),
            system_prompt=NUTRITION_SYSTEM_PROMPT,
            json_schema=NUTRITION_JSON_SCHEMA,
        )
    )
    payload = json.loads(response.text)
    return NutritionTable(
        rows=tuple(tuple(str(c) for c in row) for row in payload["rows"])
    )


def extract_nutrition_multi(
    image_paths: tuple[Path, ...],
    model: str,
    *,
    extra_system_context: str = "",
) -> NutritionTable:
    """Extract a single combined nutrition table from multiple photos.

    Sends every photo to the model as a separate image attachment in
    one call; the model handles deduplication across overlapping
    views. Avoids the lossy resize+re-encode that pre-stitching
    introduces and lets the model see each crop at its native
    resolution. ``extra_system_context`` is prepended to the system
    prompt — useful for front-context augmentation that helps the
    model interpret ambiguous columns.
    """
    if not image_paths:
        raise ValueError("extract_nutrition_multi requires at least one image")
    system_prompt = (
        extra_system_context + NUTRITION_MULTI_SYSTEM_PROMPT
        if extra_system_context
        else NUTRITION_MULTI_SYSTEM_PROMPT
    )
    response = call(
        ClaudeRequest(
            prompt=NUTRITION_MULTI_PROMPT.format(n=len(image_paths)),
            model=model,
            image_paths=image_paths,
            system_prompt=system_prompt,
            json_schema=NUTRITION_JSON_SCHEMA,
        )
    )
    payload = json.loads(response.text)
    return NutritionTable(
        rows=tuple(tuple(str(c) for c in row) for row in payload["rows"])
    )


# ---------------------------------------------------------------------------
# Price tag: store-conditional UPC slug or full UPC
# ---------------------------------------------------------------------------

PRICE_TAG_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "price_cents": {"type": "integer"},
            "description": {"type": "string"},
            "size": {"type": "string"},
            "upc": {"type": "string"},
        },
        "required": ["price_cents", "description", "size", "upc"],
        "additionalProperties": False,
    }
)

# MOM tags carry a scannable UPC-A barcode (12 digits) that equals the
# bottle's manufacturer barcode — the strongest cross-check we have.
PRICE_TAG_MOM_SYSTEM_PROMPT = (
    "You extract fields from a Mom's Organic Market shelf price tag for "
    "a deterministic comparison task. Return a JSON object with: "
    '`"price_cents"` (the regular shelf price in whole cents — e.g. '
    '$15.99 → 1599); `"description"` (the abbreviated product name on the '
    "tag, lower-cased — e.g. `nc lqud mltvitmn orng mng 30 fz`); "
    '`"size"` (the size or count printed on the tag, e.g. `30 fz`, '
    '`120 cnt`; empty string if not present); and `"upc"` (the 12-digit '
    "UPC-A code printed under the scannable barcode at the bottom of the "
    "tag — digits only, no spaces). If the UPC is unreadable, return an "
    "empty string rather than guessing. Return JSON only, no prose."
)

PRICE_TAG_MOM_PROMPT = (
    "Extract the price, description, size, and full UPC from this Mom's "
    "Organic Market shelf price tag, following the instructions in the "
    "system prompt."
)

# CVS tags carry an internal-SKU barcode (not a UPC). The header line
# `I3* DDDDD MM-DD-YY` contains a 5-digit slug that matches positions
# 6–10 of the bottle's UPC-A — the actual cross-check.
PRICE_TAG_CVS_SYSTEM_PROMPT = (
    "You extract fields from a CVS shelf price tag for a deterministic "
    "comparison task. Return a JSON object with: "
    '`"price_cents"` (the regular shelf price in whole cents — e.g. '
    '$12.79 → 1279); `"description"` (the abbreviated product name on '
    "the tag, lower-cased — e.g. `nm mv him 50+`); "
    '`"size"` (the count or size code, e.g. `90ct`, `130s`; empty string '
    'if not present); and `"upc"` (the 5-digit slug from the tag header '
    "line, which has the form `I3* DDDDD MM-DD-YY` in small text in the "
    "upper-left of the tag — return only the 5 digits, no spaces). If "
    "the slug is unreadable, return an empty string rather than guessing. "
    "Do not return the long internal barcode digits — only the 5-digit "
    "slug from the I3* header. Return JSON only, no prose."
)

PRICE_TAG_CVS_PROMPT = (
    "Extract the price, description, size code, and 5-digit UPC slug "
    "from this CVS shelf price tag, following the instructions in the "
    "system prompt."
)


@dataclass(frozen=True)
class PriceTag:
    """Shelf-tag fields. ``upc`` is store-conditional: the full 12-digit
    UPC for MOM tags, the 5-digit `I3*` header slug for CVS tags. Empty
    string when the field is absent or unreadable.
    """

    store: str  # "MOM" or "CVS"
    price_cents: int
    description: str
    size: str
    upc: str


def extract_price_tag(image_path: Path, model: str, store: str) -> PriceTag:
    """Extract shelf-tag fields, choosing the store-specific prompt.

    ``store`` is set deterministically by the caller from the photo's
    filename cutoff (Phase A decision), not by the model.
    """
    if store == "MOM":
        system_prompt = PRICE_TAG_MOM_SYSTEM_PROMPT
        prompt = PRICE_TAG_MOM_PROMPT
    elif store == "CVS":
        system_prompt = PRICE_TAG_CVS_SYSTEM_PROMPT
        prompt = PRICE_TAG_CVS_PROMPT
    else:
        raise ValueError(f"Unknown store {store!r}; expected 'MOM' or 'CVS'")

    response = call(
        ClaudeRequest(
            prompt=prompt,
            model=model,
            image_paths=(image_path,),
            system_prompt=system_prompt,
            json_schema=PRICE_TAG_JSON_SCHEMA,
        )
    )
    payload = json.loads(response.text)
    return PriceTag(
        store=store,
        price_cents=int(payload["price_cents"]),
        description=str(payload["description"]),
        size=str(payload["size"]),
        upc=str(payload["upc"]),
    )
