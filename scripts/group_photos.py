"""Walk the raw photos in chronological order and group them by product.

For each photo (after the first), we ask Sonnet 4.6 three things using the
previous photo, the current photo, and the running group's front-of-package
photo as visual context:

* Is the current photo part of the same product as the previous one?
* What role(s) does it play (front / nutrition / ingredients / price-tag /
  other-label / other)?
* A short rationale (kept in the JSON for human review).

Store assignment is deterministic from the filename cutoff
``PXL_20260426_180942709.MP.jpg`` — everything at or after is CVS, everything
before is Mom's Organic Market.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path

from _claude import ClaudeRequest, call
from _image_ops import resize_to_longest_side

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PHOTOS_DIR = REPO_ROOT / "data" / "raw_photos"
GROUPS_JSON = REPO_ROOT / "data" / "groups.json"
GROUPING_RESIZE_DIR = REPO_ROOT / "data" / "cache" / "resized_grouping"

CVS_CUTOFF_FILENAME = "PXL_20260426_180942709.MP.jpg"

VALID_ROLES = (
    "front",
    "nutrition",
    "ingredients",
    "other-label",
    "price-tag",
    "other",
)

# Sonnet doesn't need the full 4032×3024 to answer "same product?" — 1024 px
# is plenty for layout-level discrimination and keeps the per-call token cost
# bounded.
GROUPING_LONGEST_SIDE = 1024

GROUPING_SYSTEM_PROMPT = (
    "You help organize a stream of photos taken in supplement-store aisles. "
    "Each call you receive up to three images: image 1 is the IMMEDIATELY "
    "PREVIOUS photo, image 2 is the CURRENT photo (the one you must "
    "classify), and (when present) image 3 is the FRONT of the running "
    "product group. Decide whether the current photo is part of the same "
    "product as the previous one, and what role(s) it serves.\n"
    "\n"
    "SAME-PRODUCT GUIDANCE — a single product is photographed from many "
    "angles: front face, back/side panels, top, sometimes rotated 90° so "
    "the wrap-around supplement facts panel reads horizontally. The same "
    "BOTTLE may look very different from front to back (different colors, "
    "no brand logo, sideways text) but it is still the SAME product as "
    "the previous photo if: the bottle shape/color matches, the lid "
    "matches, the shelf background is the same, or you can read the same "
    "brand name even faintly. When in doubt about a back/side shot taken "
    "seconds after a clear front, treat it as the SAME product. Only "
    "declare a new product when you can see the front of a clearly "
    "different bottle (different brand, different size, different "
    "packaging colors) or a price tag for an obviously different SKU.\n"
    "\n"
    "PRIMARY SUBJECT RULE — apply roles ONLY to what the photographer is "
    "documenting in the foreground (the bottle being held, the price tag "
    "being framed, the panel being inspected). Background products on "
    "adjacent shelves, partial product fronts behind the subject, and "
    "neighbouring price tags are NOT to be labeled. If a back-of-bottle "
    "shot happens to include a sliver of an unrelated price tag in the "
    "background, do NOT apply `price-tag`.\n"
    "\n"
    "MUTUAL-EXCLUSION GUIDANCE — `front` and `nutrition` rarely co-occur "
    "on one shot (different sides of the bottle). `front` and `price-tag` "
    "essentially never co-occur (a shelf price tag is a separate object "
    "from the package). If you find yourself applying both, double-check "
    "that the second isn't a background object.\n"
    "\n"
    "ROLE DEFINITIONS:\n"
    "- `front`        — the photo's PRIMARY SUBJECT is the front face of "
    "                   a product package: brand logo, product name, and "
    "                   marketing copy fill most of the frame. Bottles "
    "                   visible behind the subject do not count.\n"
    "- `nutrition`    — ANY readable Supplement Facts / Nutrition Facts "
    "                   entries are visible on the documented panel — "
    "                   even a partial wrap-around column with a few "
    "                   vitamin/mineral rows counts. The full table need "
    "                   NOT fill the frame; this label is meant to be "
    "                   liberally applied wherever any nutrition data "
    "                   is legible.\n"
    "- `price-tag`    — the photo's PRIMARY SUBJECT is a store shelf "
    "                   price tag — a small flat tag mounted to a shelf "
    "                   edge, with a price and an abbreviated product "
    "                   name. Apply only when the tag dominates the "
    "                   frame; tags glimpsed past the subject's edge are "
    "                   ignored.\n"
    "- `ingredients`  — an ingredients list is visible on the documented "
    "                   panel ('Other Ingredients:', 'Organic ... Blend:', "
    "                   'Contains:'). Best-effort.\n"
    "- `other-label`  — other label content on the documented panel "
    "                   (warnings, lot/expiry codes, distributor info, "
    "                   side-panel marketing). Best-effort.\n"
    "- `other`        — none of the above (accidental shot, blurry floor, "
    "                   hand only, transitional shelf overview).\n"
    "\n"
    "Semantics: `front` and `price-tag` use the strict primary-subject "
    "rule (presence asserts primary, absence asserts not-primary). "
    "`nutrition` is liberal — apply whenever any nutrition entries are "
    "readable, regardless of how much of the table is visible. "
    "`ingredients` and `other-label` are best-effort — presence asserts "
    "visibility, absence is ambiguous."
)

GROUPING_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "is_same_product": {"type": "boolean"},
            "roles": {
                "type": "array",
                "items": {"type": "string", "enum": list(VALID_ROLES)},
                "minItems": 1,
            },
            "rationale": {"type": "string"},
        },
        "required": ["is_same_product", "roles", "rationale"],
        "additionalProperties": False,
    }
)


@dataclasses.dataclass
class PhotoEntry:
    path: str
    roles: list[str]


@dataclasses.dataclass
class Group:
    id: str
    store: str  # "MOM" | "CVS"
    photos: list[PhotoEntry]
    warnings: list[str] = dataclasses.field(default_factory=list)
    locked: bool = False


def store_for(filename: str) -> str:
    return "CVS" if filename >= CVS_CUTOFF_FILENAME else "MOM"


def list_photos(photo_dir: Path | None = None) -> list[Path]:
    base = photo_dir if photo_dir is not None else RAW_PHOTOS_DIR
    return sorted(base.glob("PXL_*.jpg"))


def make_group_id(store: str, sequence_in_store: int) -> str:
    short = store.lower()
    return f"20260426_{short}_{sequence_in_store:03d}"


def classify_photo(
    *,
    current: Path,
    previous: Path | None,
    front: Path | None,
    model: str = "sonnet",
) -> dict[str, object]:
    """Ask Sonnet to classify the current photo. Returns the parsed JSON dict."""
    images: list[Path] = []
    image_descriptions: list[str] = []
    if previous is not None:
        images.append(_resize_for_grouping(previous))
        image_descriptions.append(
            "Image 1 (previous photo): " + str(previous.name)
        )
    images.append(_resize_for_grouping(current))
    image_descriptions.append(
        f"Image {len(images)} (CURRENT photo): " + str(current.name)
    )
    if front is not None and front != previous:
        images.append(_resize_for_grouping(front))
        image_descriptions.append(
            f"Image {len(images)} (front of running group): " + str(front.name)
        )

    prompt = (
        "\n".join(image_descriptions)
        + "\n\n"
        + "Classify the CURRENT photo. Return JSON with keys "
        + "`is_same_product`, `roles` (one or more of "
        + ", ".join(f"`{r}`" for r in VALID_ROLES)
        + "), and a short `rationale`."
    )

    response = call(
        ClaudeRequest(
            prompt=prompt,
            model=model,
            image_paths=tuple(images),
            system_prompt=GROUPING_SYSTEM_PROMPT,
            json_schema=GROUPING_JSON_SCHEMA,
        )
    )
    return _parse_classification(response.text)


def assign_groups(
    photos: Iterable[Path],
    model: str = "sonnet",
    *,
    extension_pool: Iterable[Path] | None = None,
) -> list[Group]:
    """Walk the photos, calling Sonnet, and return the assembled groups.

    ``extension_pool``: when provided, after the last photo of ``photos`` is
    processed the classifier continues to the next photo in
    ``extension_pool`` (in iteration order) until it declares "new product",
    confirming the boundary of the last group. The boundary-confirming
    photo is NOT added to any output group — it just exists to prove the
    last group is closed. This lets a "kinks-first" sample distinguish
    "the model got the boundary right" from "the model never got asked".
    """
    photos_list = list(photos)
    groups: list[Group] = []
    sequence: dict[str, int] = {"MOM": 0, "CVS": 0}
    seen: set[str] = set()

    previous: Path | None = None

    def consume(photo: Path) -> bool:
        """Process one photo; return True iff it started a new group."""
        nonlocal previous
        store = store_for(photo.name)
        front = (
            Path(groups[-1].photos[0].path)
            if groups and groups[-1].store == store
            else None
        )
        if previous is None:
            classification: dict[str, object] = {
                "is_same_product": False,
                "roles": ["front"],
                "rationale": "First photo of the session — assumed to be a new product front.",
            }
        else:
            classification = classify_photo(
                current=photo, previous=previous, front=front, model=model
            )

        is_same = bool(classification["is_same_product"])
        roles = list(classification["roles"])  # type: ignore[arg-type]
        store_changed = (
            previous is not None and store_for(previous.name) != store
        )
        previous = photo
        if is_same and not store_changed and groups:
            groups[-1].photos.append(PhotoEntry(path=str(photo), roles=roles))
            return False
        sequence[store] += 1
        groups.append(
            Group(
                id=make_group_id(store, sequence[store]),
                store=store,
                photos=[PhotoEntry(path=str(photo), roles=roles)],
            )
        )
        return True

    for photo in photos_list:
        seen.add(str(photo))
        consume(photo)

    if extension_pool is not None and previous is not None:
        for photo in extension_pool:
            if str(photo) in seen:
                continue
            if photo.name <= previous.name:
                continue
            started_new = consume(photo)
            if started_new:
                # Boundary confirmed: drop the throwaway single-photo group.
                groups.pop()
                break

    for g in groups:
        g.warnings = compute_group_warnings(g)
    return groups


def compute_group_warnings(group: Group) -> list[str]:
    """Per-group post-checks the user can scan for in the review UI."""
    warnings: list[str] = []
    flat_roles = {r for p in group.photos for r in p.roles}
    if "front" not in flat_roles:
        warnings.append("missing front-of-package photo")
    if "nutrition" not in flat_roles:
        warnings.append("missing nutrition-facts photo")
    if "price-tag" not in flat_roles:
        warnings.append("missing shelf price tag")
    return warnings


def write_groups(groups: list[Group], out_path: Path = GROUPS_JSON) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"groups": [dataclasses.asdict(g) for g in groups]}
    out_path.write_text(json.dumps(payload, indent=2))


def _resize_for_grouping(photo: Path) -> Path:
    return resize_to_longest_side(
        photo, GROUPING_LONGEST_SIDE, GROUPING_RESIZE_DIR
    )


def _parse_classification(raw: str) -> dict[str, object]:
    cleaned = _extract_json_object(raw)
    data: dict[str, object] = json.loads(cleaned)
    is_same = data.get("is_same_product")
    roles = data.get("roles")
    rationale = data.get("rationale", "")
    if not isinstance(is_same, bool):
        raise ValueError(
            f"Expected boolean is_same_product, got {is_same!r} from {raw!r}"
        )
    if (
        not isinstance(roles, list)
        or not all(isinstance(r, str) and r in VALID_ROLES for r in roles)
        or not roles
    ):
        raise ValueError(
            f"Expected non-empty list of valid roles, got {roles!r} from {raw!r}"
        )
    return {
        "is_same_product": is_same,
        "roles": roles,
        "rationale": str(rationale),
    }


def _extract_json_object(raw: str) -> str:
    """Pull the outermost JSON object out of ``raw``. Tolerates fenced blocks."""
    import re

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            f"No JSON object found in classification response: {raw!r}"
        )
    return raw[start : end + 1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Group raw photos by product.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N photos (for kinks-first runs). When "
            "set, the classifier still continues PAST the limit on demand "
            "until it sees a 'new product' transition, so the last group's "
            "boundary is decided by the model rather than by the cap."
        ),
    )
    parser.add_argument(
        "--model",
        default="sonnet",
        help="Model alias passed to claude -p (haiku/sonnet/opus).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=GROUPS_JSON,
        help="Output groups.json path.",
    )
    args = parser.parse_args()

    all_photos = list_photos()
    if args.limit is not None:
        sample = all_photos[: args.limit]
        extension: list[Path] = all_photos[args.limit :]
    else:
        sample = all_photos
        extension = []
    groups = assign_groups(
        sample, model=args.model, extension_pool=extension or None
    )
    write_groups(groups, out_path=args.out)
    print(
        f"Wrote {len(groups)} groups across {len(sample)} sampled photos "
        f"(plus boundary-resolution probes) to {args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
