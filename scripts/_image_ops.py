"""Image-resize helpers used by the sizing finder and stitching stages."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def resize_to_longest_side(
    src: Path, longest_side: int, cache_dir: Path
) -> Path:
    """Resize ``src`` so its longest dimension is ``longest_side`` px.

    EXIF rotation is applied first. Output is JPEG quality 85, written to
    ``{cache_dir}/{src.stem}_{longest_side}.jpg``. Idempotent: returns the
    cached path on a re-run without re-encoding.
    """
    if longest_side < 1:
        raise ValueError(f"longest_side must be >= 1, got {longest_side}")
    cache_path = cache_dir / f"{src.stem}_{longest_side}.jpg"
    if cache_path.exists():
        return cache_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        if (
            img is None
        ):  # pragma: no cover — exif_transpose only returns None for None input
            raise RuntimeError(f"exif_transpose returned None for {src}")
        img.thumbnail((longest_side, longest_side), Image.Resampling.LANCZOS)
        img.convert("RGB").save(cache_path, "JPEG", quality=85)
    return cache_path


def longest_side_px(path: Path) -> int:
    """Return the longest side (px) of the image at ``path``, EXIF-corrected."""
    with Image.open(path) as img:
        oriented = ImageOps.exif_transpose(img)
        if oriented is None:  # pragma: no cover
            raise RuntimeError(f"exif_transpose returned None for {path}")
        return max(oriented.size)
