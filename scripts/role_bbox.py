"""Per-role bounding-box detection and cropping.

A small haiku call identifies every text-bearing panel in a product
photo, returning a tight fractional bbox plus the natural reading
direction of the panel ("horizontal" or "vertical"). The reading
direction is per-panel because the user sometimes turns a bottle on its
side and sometimes doesn't — direction isn't deterministic by role.

The cropped panels become the inputs to per-role extraction
(:mod:`role_extraction`); the reading direction tells the per-role
stitcher which axis to concatenate along (next stage).

Role taxonomy mirrors :mod:`group_photos` exactly so a photo's
``roles`` field and the detected panels can be cross-referenced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from _claude import ClaudeRequest, call
from _image_ops import resize_to_longest_side
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
CROP_CACHE_DIR = REPO_ROOT / "data" / "cache" / "cropped_panels"

ROLE_KINDS = ("front", "nutrition", "ingredients", "other-label", "price-tag")
TEXT_DIRECTIONS = ("horizontal", "vertical")

BBOX_SYSTEM_PROMPT = (
    "You locate text-bearing labels on the PRIMARY product in a "
    "product photo. The primary product is the one the photo is "
    "focused on: typically held by a hand, in the foreground, "
    "largest, sharpest, and centered. Ignore everything else in the "
    "frame: products on adjacent shelves, products behind or beside "
    "the primary product, and partial / out-of-focus products at the "
    "edges. A shelf price tag belongs to the primary product only if "
    "it is the directly attached / adjacent tag for that exact "
    "product — tags belonging to other shelf items are NOT primary. "
    "Return JSON only, no prose."
)


def _bbox_prompt(expected_roles: tuple[str, ...]) -> str:
    """Build the prompt with role choices restricted to the expected set.

    Restricting the choices to roles the caller knows the photo can
    contain (via position-derived rules, e.g. front=first, price-tag=
    last) eliminates the failure mode where the detector reports
    background products' panels — the model literally cannot emit an
    off-target role since the schema enum doesn't include it.
    """
    roles_str = ", ".join(repr(k) for k in expected_roles)
    return (
        "Identify the text-bearing labels and panels of the PRIMARY product "
        "in this photo. Do not report panels belonging to other products on "
        "adjacent shelves or in the background — only the primary subject. "
        "If the photo is just a shelf price tag, the primary product IS "
        "that price tag. Look ONLY for these role kinds: "
        f"{roles_str}. Skip any other panels. "
        "For each panel return: "
        f'`"kind"` — one of {roles_str}; '
        '`"x_min_frac"`, `"y_min_frac"`, `"x_max_frac"`, `"y_max_frac"` — '
        "TIGHT bounding-box coordinates (fractions in [0, 1], (0,0) "
        "top-left, (1,1) bottom-right) cropped close to the printed text "
        "only. Exclude bottle curvature, specular glare, hands, shelf, "
        "and background — at most a few percent margin past the readable "
        "text on each side; and "
        '`"text_direction"` — `"horizontal"` if the text reads '
        'left-to-right as the photo is oriented, or `"vertical"` if it '
        "reads top-to-bottom (e.g. text rotated 90 degrees because the "
        "package is on its side). Return JSON only."
    )


def _bbox_json_schema(expected_roles: tuple[str, ...]) -> str:
    """Schema with the kind enum restricted to ``expected_roles``."""
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "panels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": list(expected_roles),
                            },
                            "x_min_frac": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "y_min_frac": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "x_max_frac": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "y_max_frac": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "text_direction": {
                                "type": "string",
                                "enum": list(TEXT_DIRECTIONS),
                            },
                        },
                        "required": [
                            "kind",
                            "x_min_frac",
                            "y_min_frac",
                            "x_max_frac",
                            "y_max_frac",
                            "text_direction",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["panels"],
            "additionalProperties": False,
        }
    )


@dataclass(frozen=True)
class PanelBbox:
    """One detected panel: role kind + fractional bbox + reading direction."""

    kind: str
    x_min_frac: float
    y_min_frac: float
    x_max_frac: float
    y_max_frac: float
    text_direction: str

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert fractional coords to integer pixel box (left, top, right, bot)."""
        return (
            int(self.x_min_frac * width),
            int(self.y_min_frac * height),
            int(self.x_max_frac * width),
            int(self.y_max_frac * height),
        )


def detect_panels(
    image_path: Path,
    *,
    expected_roles: tuple[str, ...],
    model: str = "haiku",
    detect_at_longest_side: int = 1024,
) -> tuple[PanelBbox, ...]:
    """Run the bbox detector at a small resize and return per-panel boxes.

    ``expected_roles`` is required: the caller declares which roles the
    photo is expected to contain (front/nutrition/price-tag are
    deterministic from group position; loose roles can be passed when
    relevant). The schema enum is restricted to ``expected_roles`` so
    the model cannot emit off-target kinds even if it sees background
    products' panels.

    The detector runs at ``detect_at_longest_side`` (default 1024 px) —
    bbox detection doesn't need full resolution and the smaller image
    cuts the call cost without measurably hurting box accuracy.
    """
    if not expected_roles:
        raise ValueError("expected_roles must be non-empty")
    if any(r not in ROLE_KINDS for r in expected_roles):
        bad = [r for r in expected_roles if r not in ROLE_KINDS]
        raise ValueError(
            f"expected_roles contains unknown kind(s) {bad}; valid: {ROLE_KINDS}"
        )
    detect_path = resize_to_longest_side(
        image_path, detect_at_longest_side, CROP_CACHE_DIR / "detect_inputs"
    )
    response = call(
        ClaudeRequest(
            prompt=_bbox_prompt(expected_roles),
            model=model,
            image_paths=(detect_path,),
            system_prompt=BBOX_SYSTEM_PROMPT,
            json_schema=_bbox_json_schema(expected_roles),
        )
    )
    payload = json.loads(response.text)
    return tuple(
        PanelBbox(
            kind=str(p["kind"]),
            x_min_frac=float(p["x_min_frac"]),
            y_min_frac=float(p["y_min_frac"]),
            x_max_frac=float(p["x_max_frac"]),
            y_max_frac=float(p["y_max_frac"]),
            text_direction=str(p["text_direction"]),
        )
        for p in payload["panels"]
    )


def crop_panel(
    image_path: Path,
    panel: PanelBbox,
    *,
    out_dir: Path = CROP_CACHE_DIR,
) -> Path:
    """EXIF-transpose, crop to ``panel``, save as JPEG, return the path.

    Crop output is keyed by source filename + panel coordinates so a
    re-run on the same input is idempotent — the cropped file appears
    once and downstream extractions can rely on its path being stable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    coord_tag = (
        f"{int(panel.x_min_frac * 1000):04d}"
        f"_{int(panel.y_min_frac * 1000):04d}"
        f"_{int(panel.x_max_frac * 1000):04d}"
        f"_{int(panel.y_max_frac * 1000):04d}"
    )
    out_path = out_dir / f"{image_path.stem}__{panel.kind}__{coord_tag}.jpg"
    if out_path.exists():
        return out_path

    with Image.open(image_path) as im:
        oriented = ImageOps.exif_transpose(im)
        if oriented is None:  # pragma: no cover — only None for None input
            oriented = im.copy()
        try:
            box = panel.to_pixels(oriented.width, oriented.height)
            cropped = oriented.crop(box)
            try:
                cropped.convert("RGB").save(out_path, "JPEG", quality=92)
            finally:
                cropped.close()
        finally:
            oriented.close()
    return out_path
