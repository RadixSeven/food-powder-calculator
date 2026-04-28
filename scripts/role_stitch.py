"""Per-role stitching of cropped panels along the text-reading axis.

Replaces the all-photos-in-one-vertical-stack design from
:mod:`stitch` (Phase A draft) with a role-aware stitcher: panels of the
same role from one group are concatenated along the axis that matches
the text-reading direction.

* ``"horizontal"`` text → stitch left-to-right, so the next frame
  continues the text where the previous one left off (the wrap-around
  use case).
* ``"vertical"`` text → stitch top-to-bottom (the bottle was laid on
  its side at capture time).

Capture-order sequencing is the caller's responsibility: pass panel
paths in the order the user photographed them, and the stitch direction
matches the rotation/text flow naturally.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from role_bbox import PanelBbox

# Thin separator strip between panels so the model has an unambiguous
# visual boundary between adjacent frames.
SEPARATOR_PX = 4
SEPARATOR_COLOR = (200, 200, 200)

# Cap the output's longest side so Claude's vision pipeline doesn't
# downsample uncontrollably. Multi-shot stitches can balloon to
# 10000+ px on the long axis when 3-5 photos at 3000px each are
# concatenated; the API then downsamples internally and we lose
# detail in unpredictable ways. Capping the output here lets us
# control the resampling — LANCZOS at our quality setting beats
# whatever the server-side resize uses.
DEFAULT_MAX_OUTPUT_LONGEST_SIDE = 4096


def stitch_axis_for_panels(panels: list[PanelBbox]) -> str:
    """Pick the stitch axis from the majority ``text_direction``.

    Within one group + role the panels usually agree on direction
    because they're photos of the same continuous label. If a group
    happens to mix orientations (rare — user rotated the bottle between
    frames), the majority wins and ties break toward ``"horizontal"``
    since that's the more common case.
    """
    if not panels:
        raise ValueError("stitch_axis_for_panels requires at least one panel")
    counts = Counter(p.text_direction for p in panels)
    horiz = counts.get("horizontal", 0)
    vert = counts.get("vertical", 0)
    return "horizontal" if horiz >= vert else "vertical"


def stitch_role_panels(
    panel_paths: list[Path],
    text_direction: str,
    out_path: Path,
    *,
    match_scale: bool = True,
    max_output_longest_side: int | None = DEFAULT_MAX_OUTPUT_LONGEST_SIDE,
) -> None:
    """Stitch ``panel_paths`` along ``text_direction`` and write JPEG.

    ``text_direction`` must be ``"horizontal"`` or ``"vertical"``.
    Panels are loaded in the order given (caller sorts by capture time).
    Each panel is centered along the cross-axis on a separator-color
    canvas — this is how a shorter frame next to a taller one stays
    visually anchored.

    When ``match_scale`` is True (the default), every panel is
    resized so its longest-side matches the LARGEST panel's longest-
    side. This eliminates the multi-scale issue where one frame is
    naturally 3000px and another is 800px — without matching, opus
    reads the larger ones at high detail and the smaller ones as
    illegible blurs. Matching to MAX (rather than min) preserves the
    detail in the largest panel; smaller panels just upsample to fill
    the canvas without information gain — but also without losing
    detail in the panels that have it. Set False to preserve raw crop
    sizes (useful for debugging or when sizes are already matched).
    """
    if not panel_paths:
        raise ValueError("stitch_role_panels requires at least one panel")
    if text_direction not in ("horizontal", "vertical"):
        raise ValueError(
            f"text_direction must be 'horizontal' or 'vertical', "
            f"got {text_direction!r}"
        )
    panels: list[Image.Image] = []
    try:
        for p in panel_paths:
            panels.append(Image.open(p).convert("RGB"))

        if match_scale and len(panels) > 1:
            target_longest = max(max(img.size) for img in panels)
            panels = _resize_to_longest(panels, target_longest)

        if text_direction == "horizontal":
            canvas = _stitch_horizontal(panels)
        else:
            canvas = _stitch_vertical(panels)

        if (
            max_output_longest_side is not None
            and max(canvas.size) > max_output_longest_side
        ):
            ratio = max_output_longest_side / max(canvas.size)
            new_size = (
                int(canvas.width * ratio),
                int(canvas.height * ratio),
            )
            shrunk = canvas.resize(new_size, Image.Resampling.LANCZOS)
            canvas.close()
            canvas = shrunk

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            canvas.save(out_path, "JPEG", quality=85)
        finally:
            canvas.close()
    finally:
        for img in panels:
            img.close()


def _resize_to_longest(
    panels: list[Image.Image], target_longest: int
) -> list[Image.Image]:
    """Resize each panel so its longest side equals ``target_longest``.

    Panels already at that longest-side are returned unchanged; the
    rest get LANCZOS-resampled proportional resize (up or down,
    depending on whether their longest-side is below or above the
    target). Original images are closed and replaced with the
    resized copies; the caller still owns the returned list and is
    responsible for closing them.
    """
    out: list[Image.Image] = []
    for img in panels:
        longest = max(img.size)
        if longest == target_longest:
            out.append(img)
            continue
        ratio = target_longest / longest
        new_size = (int(img.width * ratio), int(img.height * ratio))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        img.close()
        out.append(resized)
    return out


def _stitch_horizontal(panels: list[Image.Image]) -> Image.Image:
    """Left-to-right concat; canvas height = max panel height."""
    max_height = max(img.height for img in panels)
    total_width = sum(img.width for img in panels)
    if len(panels) > 1:
        total_width += SEPARATOR_PX * (len(panels) - 1)
    canvas = Image.new("RGB", (total_width, max_height), SEPARATOR_COLOR)
    x = 0
    for i, img in enumerate(panels):
        if i > 0:
            x += SEPARATOR_PX
        y = (max_height - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width
    return canvas


def _stitch_vertical(panels: list[Image.Image]) -> Image.Image:
    """Top-to-bottom concat; canvas width = max panel width."""
    max_width = max(img.width for img in panels)
    total_height = sum(img.height for img in panels)
    if len(panels) > 1:
        total_height += SEPARATOR_PX * (len(panels) - 1)
    canvas = Image.new("RGB", (max_width, total_height), SEPARATOR_COLOR)
    y = 0
    for i, img in enumerate(panels):
        if i > 0:
            y += SEPARATOR_PX
        x = (max_width - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height
    return canvas
