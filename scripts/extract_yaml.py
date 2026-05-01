"""Run role-specific extraction on per-group stitched panels.

Consumes the artifacts from :mod:`group_pipeline` (one image per role
per group) and produces one structured YAML file per group with the
combined per-role extraction output.

Roles handled:

* ``front``: product name + manufacturer (haiku is enough — the
  panel is large, simple text).
* ``nutrition``: full table extraction (opus — the table is dense
  and column ordering matters for downstream codegen).
* ``price-tag``: store-conditional price/UPC fields (haiku — the
  fields are short and structured).

The front extraction runs first; its product name is appended to the
nutrition system prompt as context, so the model can disambiguate
multi-column tables (kid-vs-adult %DV columns are easier to assign
when "Children's Liquid Multivitamin Ages 2-13" is in scope).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

import group_pipeline
from _claude import ClaudeRequest, call
from _pipeline import pipeline_step, register_producer
from _review_crops import DEFAULT_CROP_REVIEWS_JSON, crops_flagged_info_lost
from _run import open_run
from group_pipeline import (
    GOLD_GROUPS_JSON,
    REPO_ROOT,
    STITCHED_PANELS_DIR,
    GroupArtifacts,
    detect_and_crop_group,
    load_group,
    stitch_role_outputs,
)
from role_extraction import (
    NUTRITION_JSON_SCHEMA,
    NUTRITION_PROMPT,
    NUTRITION_SYSTEM_PROMPT,
    FrontExtraction,
    NutritionTable,
    PriceTag,
    extract_front,
    extract_nutrition,
    extract_nutrition_multi,
    extract_price_tag,
)

EXTRACTED_YAML_DIR = REPO_ROOT / "data" / "extracted_yaml"

NUTRITION_MODEL = "opus"
FRONT_MODEL = "haiku"
PRICE_TAG_MODEL = "haiku"

# Producer registration: extracted YAMLs are produced by this script.
register_producer(
    "data/extracted_yaml/", "uv run python scripts/extract_yaml.py <gid>"
)


@dataclass(frozen=True)
class GroupExtraction:
    """Combined per-role extractions for one group."""

    group_id: str
    store: str
    front: FrontExtraction | None
    nutrition: NutritionTable | None
    price_tag: PriceTag | None


def extract_group(
    artifacts: GroupArtifacts,
    *,
    crop_reviews_json: Path = DEFAULT_CROP_REVIEWS_JSON,
) -> GroupExtraction:
    """Run role-specific extraction on each per-role artifact.

    For nutrition, any crop flagged "info lost" in the crop review
    state (✂ info_excluded with no recovery pointer) gets its
    uncropped source photo added to the image set. The model sees
    both the cropped panel and the wider context, so text the bbox
    dropped can be recovered from the original photo. Less
    information-dense than a tight crop, more tokens, but recovers
    text we'd otherwise be missing.
    """
    front: FrontExtraction | None = None
    if "front" in artifacts.panels_by_role:
        print(
            f"  [extract] {artifacts.group_id}: front",
            file=sys.stderr,
            flush=True,
        )
        front = extract_front(artifacts.panels_by_role["front"], FRONT_MODEL)

    nutrition: NutritionTable | None = None
    if "nutrition" in artifacts.crops_by_role:
        crops = artifacts.crops_by_role["nutrition"]
        lost_filenames = crops_flagged_info_lost(
            crop_reviews_json, artifacts.group_id
        )
        # Add the uncropped source photo for every nutrition crop the
        # reviewer marked "info lost". Order: all crops first, then
        # the appended uncropped sources. Dedup is the model's job
        # (system prompt covers it).
        uncropped_sources = tuple(
            artifacts.crop_sources[crop]
            for crop in crops
            if crop.name in lost_filenames
        )
        n_uncropped = len(uncropped_sources)
        suffix = (
            f" + {n_uncropped} uncropped source"
            f"{'s' if n_uncropped > 1 else ''} (info lost)"
            if n_uncropped
            else ""
        )
        print(
            f"  [extract] {artifacts.group_id}: nutrition "
            f"({len(crops)} crop{'s' if len(crops) > 1 else ''}{suffix})",
            file=sys.stderr,
            flush=True,
        )
        nutrition = _extract_nutrition_with_front_context(
            crops + uncropped_sources, front
        )

    price_tag: PriceTag | None = None
    if "price-tag" in artifacts.panels_by_role:
        print(
            f"  [extract] {artifacts.group_id}: price-tag ({artifacts.store})",
            file=sys.stderr,
            flush=True,
        )
        price_tag = extract_price_tag(
            artifacts.panels_by_role["price-tag"],
            PRICE_TAG_MODEL,
            store=artifacts.store,
        )

    return GroupExtraction(
        group_id=artifacts.group_id,
        store=artifacts.store,
        front=front,
        nutrition=nutrition,
        price_tag=price_tag,
    )


def _extract_nutrition_with_front_context(
    image_paths: tuple[Path, ...], front: FrontExtraction | None
) -> NutritionTable:
    """Run nutrition extraction with optional front-text context.

    Single-image path goes through :func:`extract_nutrition`;
    multi-image path goes through :func:`extract_nutrition_multi`,
    which sends every crop as a separate attachment so the model
    sees each at native resolution and handles deduplication
    itself — no stitch artifacts.

    Front product name is prepended to the system prompt as context
    so the model can disambiguate ambiguous columns (e.g. age-band
    %DV columns). When front extraction failed, falls back to the
    bare extractor.
    """
    context = ""
    if front is not None:
        context = (
            "Context: these images show the nutrition / supplement "
            f"facts panel for the product {front.product_name!r}"
            + (f" by {front.manufacturer!r}" if front.manufacturer else "")
            + ". Use that to disambiguate column headers (e.g. "
            "age-band %DV columns) when reading the table.\n\n"
        )
    if len(image_paths) == 1:
        if front is None:
            return extract_nutrition(image_paths[0], NUTRITION_MODEL)
        return _extract_nutrition_single_with_context(image_paths[0], context)
    return extract_nutrition_multi(
        image_paths, NUTRITION_MODEL, extra_system_context=context
    )


def _extract_nutrition_single_with_context(
    image_path: Path, context: str
) -> NutritionTable:
    """Single-image nutrition extraction with a context preamble."""
    response = call(
        ClaudeRequest(
            prompt=NUTRITION_PROMPT,
            model=NUTRITION_MODEL,
            image_paths=(image_path,),
            system_prompt=context + NUTRITION_SYSTEM_PROMPT,
            json_schema=NUTRITION_JSON_SCHEMA,
        )
    )
    payload = json.loads(response.text)
    return NutritionTable(
        rows=tuple(tuple(str(c) for c in row) for row in payload["rows"])
    )


def write_extraction_yaml(extraction: GroupExtraction, out_path: Path) -> None:
    """Serialize the extraction to YAML for human review and codegen."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "group_id": extraction.group_id,
        "store": extraction.store,
    }
    if extraction.front is not None:
        payload["front"] = dataclasses.asdict(extraction.front)
    if extraction.nutrition is not None:
        payload["nutrition"] = {
            "rows": [list(row) for row in extraction.nutrition.rows]
        }
    if extraction.price_tag is not None:
        payload["price_tag"] = dataclasses.asdict(extraction.price_tag)
    with out_path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _process_group_to_yaml_inputs(
    group_id: str,
    *,
    gold_path: Path = GOLD_GROUPS_JSON,
    stitched_root: Path = STITCHED_PANELS_DIR,
    out_dir: Path = EXTRACTED_YAML_DIR,
    skip_if_exists: bool = True,
) -> tuple[Path, ...]:
    """Resolve the durable inputs for one process_group_to_yaml call.

    Always: ``gold_path``. Additionally, every photo the gold entry
    for ``group_id`` references — but only when ``gold_path`` exists,
    so the wrapper's missing-gold error fires first instead of a deeper
    KeyError from ``load_group``. Plus the crop-review state file when
    it exists, since "info lost" flags there change which images get
    sent to the nutrition extractor.
    """
    del stitched_root, out_dir, skip_if_exists
    if not gold_path.exists():
        return (gold_path,)
    try:
        group = load_group(gold_path, group_id)
    except (KeyError, ValueError):
        return (gold_path,)
    photos = group.get("photos")
    paths: list[Path] = [gold_path]
    if isinstance(photos, list):
        for ph in photos:
            if not isinstance(ph, dict):
                continue
            path_value = ph.get("path")
            if isinstance(path_value, str):
                # Reference group_pipeline.REPO_ROOT through the module
                # rather than the module-level import so test fixtures
                # that monkeypatch ``group_pipeline.REPO_ROOT`` are
                # honored here too.
                paths.append(group_pipeline.REPO_ROOT / path_value)
    if DEFAULT_CROP_REVIEWS_JSON.exists():
        paths.append(DEFAULT_CROP_REVIEWS_JSON)
    return tuple(paths)


def _process_group_to_yaml_outputs(
    group_id: str,
    *,
    gold_path: Path = GOLD_GROUPS_JSON,
    stitched_root: Path = STITCHED_PANELS_DIR,
    out_dir: Path = EXTRACTED_YAML_DIR,
    skip_if_exists: bool = True,
) -> tuple[Path, ...]:
    """Resolve the durable output of one process_group_to_yaml call."""
    del gold_path, stitched_root, skip_if_exists
    return (out_dir / f"{group_id}.yaml",)


@pipeline_step(
    inputs=_process_group_to_yaml_inputs,
    outputs=_process_group_to_yaml_outputs,
    name="extract_yaml.process_group_to_yaml",
    # The function has a skip-when-output-exists fast path; checking
    # inputs eagerly would defeat that — a stale ``gold.json`` should
    # not block re-using an already-extracted YAML.
    eager_input_check=False,
)
def process_group_to_yaml(
    group_id: str,
    *,
    gold_path: Path = GOLD_GROUPS_JSON,
    stitched_root: Path = STITCHED_PANELS_DIR,
    out_dir: Path = EXTRACTED_YAML_DIR,
    skip_if_exists: bool = True,
) -> Path:
    """End-to-end for one group: bbox/crop/stitch + extract + write YAML.

    Returns the path of the written YAML. Idempotent at the bbox/crop
    layer (cached) and at the extraction layer (also cached). With
    ``skip_if_exists=True`` (default), groups whose YAML already
    exists are short-circuited — useful for resuming a long batch
    after a config change without redoing the cheap-to-cache work
    we've already paid for. Pass ``skip_if_exists=False`` (or delete
    the YAML beforehand) to force re-extraction.
    """
    out_path = out_dir / f"{group_id}.yaml"
    if skip_if_exists and out_path.exists():
        print(
            f"[extract_yaml] {group_id}: YAML exists, skipping",
            file=sys.stderr,
            flush=True,
        )
        return out_path
    group = load_group(gold_path, group_id)
    by_role = detect_and_crop_group(
        group, crop_dir=stitched_root / group_id / "crops"
    )
    artifacts = stitch_role_outputs(
        group_id, by_role, out_dir=stitched_root / group_id
    )
    extraction = extract_group(artifacts)
    write_extraction_yaml(extraction, out_path)
    print(
        f"[extract_yaml] {group_id}: wrote {out_path}",
        file=sys.stderr,
        flush=True,
    )
    return out_path


def main() -> int:  # pragma: no cover — CLI entry, exercised manually
    """Command-line entry: extract YAML for one or more group ids."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group_ids",
        nargs="+",
        help="One or more group IDs to process.",
    )
    parser.add_argument(
        "--gold", type=Path, default=GOLD_GROUPS_JSON, help="Gold-groups JSON."
    )
    parser.add_argument(
        "--stitched-root",
        type=Path,
        default=STITCHED_PANELS_DIR,
        help="Per-group stitched-panel output root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXTRACTED_YAML_DIR,
        help="Where to write extracted YAML files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if a YAML already exists for the group.",
    )
    args = parser.parse_args()
    failures: list[tuple[str, str]] = []
    with open_run(argv=sys.argv):
        for gid in args.group_ids:
            try:
                process_group_to_yaml(
                    gid,
                    gold_path=args.gold,
                    stitched_root=args.stitched_root,
                    out_dir=args.out_dir,
                    skip_if_exists=not args.force,
                )
            except Exception as e:  # pragma: no cover — last-line resilience
                failures.append((gid, repr(e)))
                print(
                    f"[extract_yaml] {gid}: FAILED — {e!r}; continuing",
                    file=sys.stderr,
                    flush=True,
                )
    if failures:
        print(
            f"\n[extract_yaml] {len(failures)} group(s) failed:",
            file=sys.stderr,
            flush=True,
        )
        for gid, err in failures:
            print(f"  {gid}: {err}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
