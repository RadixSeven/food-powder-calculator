"""Per-group stitched images for the YAML-extraction stage.

Each group in ``data/gold_groups.json`` becomes one stitched JPEG in
``data/stitched/<group_id>.jpg``. Each source photo is resized to its
*own* measured ``min_legible_size`` (from
``find_legible_size.binary_search_min_size``) before stitching, so a
front-of-package photo can be downsampled aggressively while a small-
print supplement-facts wrap-around stays close to original resolution.

The first full run is slow: ~10 Claude calls per photo × ~200 photos
under a 5-hour rolling rate-limit window, with the auto-retry handling
the waits. Subsequent runs reuse the per-call cache in ``data/cache/``
and complete in seconds. Per-group output is idempotent — if a stitched
file exists, the group is skipped.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from _claude import ClaudeStructuredOutputError
from _image_ops import resize_to_longest_side
from _json_types import JsonValue
from find_legible_size import (
    binary_search_min_size,
    extract_payload,
    select_search_model,
)
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_GROUPS_JSON = REPO_ROOT / "data" / "gold_groups.json"
STITCHED_DIR = REPO_ROOT / "data" / "stitched"
STITCH_RESIZE_CACHE = REPO_ROOT / "data" / "cache" / "resized_for_stitch"

# Thin separator strip between stacked photos so the model has an
# unambiguous visual boundary between panels.
SEPARATOR_HEIGHT_PX = 4
SEPARATOR_COLOR = (200, 200, 200)


@dataclasses.dataclass(frozen=True)
class SizedPhoto:
    """A source photo plus the longest-side it should be resized to."""

    path: Path
    longest_side: int


def compute_min_legible_size(photo: Path) -> int:
    """Run the per-photo binary search and return its min legible size.

    Calls go through the cached ``_claude.call`` wrapper, so re-invoking
    on the same photo replays from the cache without API cost.
    """
    reference = extract_payload(photo, "opus")
    search_model, _ = select_search_model(photo, reference)
    result = binary_search_min_size(
        photo, search_model, reference, photo_id=photo.name
    )
    return result.min_legible_size


def stitch_group(
    group_id: str,
    sized_photos: list[SizedPhoto],
    out_path: Path,
    *,
    resize_cache_dir: Path = STITCH_RESIZE_CACHE,
) -> None:
    """Vertically concatenate ``sized_photos`` and write to ``out_path``.

    Each photo is resized to its own ``longest_side`` (smaller photos are
    centered horizontally on the canvas); the canvas width is the largest
    resized width across the group. EXIF orientation is applied by
    :func:`resize_to_longest_side`.
    """
    if not sized_photos:
        raise ValueError(f"Group {group_id} has no photos to stitch")
    panels: list[Image.Image] = []
    try:
        for sp in sized_photos:
            resized = resize_to_longest_side(
                sp.path, sp.longest_side, resize_cache_dir
            )
            panels.append(Image.open(resized).convert("RGB"))

        max_width = max(img.width for img in panels)
        total_height = sum(img.height for img in panels)
        if len(panels) > 1:
            total_height += SEPARATOR_HEIGHT_PX * (len(panels) - 1)

        canvas = Image.new("RGB", (max_width, total_height), SEPARATOR_COLOR)
        y = 0
        for i, img in enumerate(panels):
            if i > 0:
                y += SEPARATOR_HEIGHT_PX
            x = (max_width - img.width) // 2
            canvas.paste(img, (x, y))
            y += img.height
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, "JPEG", quality=85)
    finally:
        for img in panels:
            img.close()


def _photos_in_group(g: dict[str, JsonValue]) -> list[Path]:
    """Repo-relative photo paths for one group, narrowed to actual strings."""
    photos = g.get("photos")
    if not isinstance(photos, list):
        return []
    out: list[Path] = []
    for p in photos:
        if not isinstance(p, dict):
            continue
        path_value = p.get("path")
        if isinstance(path_value, str):
            out.append(REPO_ROOT / path_value)
    return out


def stitch_all_groups(
    gold_groups_json: Path,
    out_dir: Path,
) -> list[Path]:
    """Iterate every group, size + stitch, return paths of stitched images."""
    payload: JsonValue = json.loads(gold_groups_json.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {gold_groups_json}")
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"Expected groups list in {gold_groups_json}")

    out_paths: list[Path] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        gid = group.get("id")
        if not isinstance(gid, str):
            continue
        out_path = out_dir / f"{gid}.jpg"
        if out_path.exists():
            print(f"[stitch] {gid}: cached", file=sys.stderr, flush=True)
            out_paths.append(out_path)
            continue
        photo_paths = _photos_in_group(group)
        if not photo_paths:
            continue
        sized: list[SizedPhoto] = []
        for photo_path in photo_paths:
            print(
                f"[stitch] {gid}: sizing {photo_path.name}",
                file=sys.stderr,
                flush=True,
            )
            try:
                longest = compute_min_legible_size(photo_path)
            except ClaudeStructuredOutputError as e:
                # Reference Opus call couldn't satisfy the schema even at
                # full resolution — typically a photo with very little text
                # or unusually busy framing. Drop the individual photo from
                # this group rather than failing the entire run.
                print(
                    f"[stitch] {gid}: SKIP {photo_path.name} "
                    f"(sizing failed: {e.failure_message})",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            sized.append(SizedPhoto(path=photo_path, longest_side=longest))
        if not sized:
            print(
                f"[stitch] {gid}: SKIP group (no photos could be sized)",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            f"[stitch] {gid}: stitching {len(sized)} photos",
            file=sys.stderr,
            flush=True,
        )
        stitch_group(gid, sized, out_path)
        out_paths.append(out_path)
    return out_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stitch one image per product group, sized per-photo."
    )
    parser.add_argument(
        "--gold-groups",
        type=Path,
        default=GOLD_GROUPS_JSON,
        help="Source groups JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=STITCHED_DIR,
        help="Directory to write stitched JPEGs into.",
    )
    args = parser.parse_args()
    out_paths = stitch_all_groups(args.gold_groups, args.out_dir)
    print(
        f"[stitch] DONE — {len(out_paths)} groups under {args.out_dir}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
