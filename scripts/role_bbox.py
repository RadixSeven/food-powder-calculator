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
    "You are an expert at locating text-bearing labels on product photos. "
    "Identify every distinct readable panel in the image and return one "
    "tight bounding box per panel. Do not invent panels that aren't "
    "clearly readable. Crop each box tight to the readable text area: "
    "exclude bottle curvature, specular glare, hands holding the package, "
    "and surrounding shelf / background. Return JSON only, no prose."
)

BBOX_PROMPT = (
    "Identify every text-bearing label or panel in this product photo. "
    "For each panel return: "
    '`"kind"` — one of '
    f"{', '.join(repr(k) for k in ROLE_KINDS)}; "
    '`"x_min_frac"`, `"y_min_frac"`, `"x_max_frac"`, `"y_max_frac"` — '
    "tight bounding-box coordinates as fractions in [0, 1] with (0,0) at "
    "the top-left and (1,1) at the bottom-right; and "
    '`"text_direction"` — `"horizontal"` if the text reads left-to-right '
    'as the photo is oriented, or `"vertical"` if it reads top-to-bottom '
    "(e.g. text rotated 90 degrees because the package is on its side). "
    "Return JSON only."
)

BBOX_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "panels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(ROLE_KINDS)},
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
    model: str = "haiku",
    detect_at_longest_side: int = 1024,
) -> tuple[PanelBbox, ...]:
    """Run the bbox detector at a small resize and return per-panel boxes.

    The detector runs at ``detect_at_longest_side`` (default 1024 px) —
    bbox detection doesn't need full resolution and the smaller image
    cuts the call cost without measurably hurting box accuracy.
    """
    detect_path = resize_to_longest_side(
        image_path, detect_at_longest_side, CROP_CACHE_DIR / "detect_inputs"
    )
    response = call(
        ClaudeRequest(
            prompt=BBOX_PROMPT,
            model=model,
            image_paths=(detect_path,),
            system_prompt=BBOX_SYSTEM_PROMPT,
            json_schema=BBOX_JSON_SCHEMA,
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
