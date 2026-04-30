"""Per-group orchestrator: position rules → bbox → crop → per-role stitch.

For one product group from ``data/gold_groups.json``, produce one
stitched image per role under ``data/stitched_panels/<group_id>/``.
The strict-role pipeline:

1. Apply position rules (verified 100% on gold across 47 groups):
   front = first photo, price-tag = last photo iff any photo has it.
2. For each photo, ask the bbox detector for *only* the roles that
   photo's gold entry says it has — the schema-restricted refactor
   means the model can't emit off-target kinds.
3. Crop each detected panel.
4. For each role across the group, stitch the crops along the
   majority text direction (or use the single crop directly if the
   group only has one shot of that role — the "no-stitch single-shot
   bypass" the user requested).

This module does no sizing — every crop is the natural per-photo size
straight out of the bbox crop. Sizing is a separate concern and lives
in :mod:`sizing_finder` (next step) for the multi-shot case where it
matters most.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from _json_types import JsonObject, JsonValue
from _pipeline import pipeline_step, register_producer
from role_bbox import ROLE_KINDS, PanelBbox, crop_panel, detect_panels
from role_stitch import stitch_axis_for_panels, stitch_role_panels

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_GROUPS_JSON = REPO_ROOT / "data" / "gold_groups.json"
STITCHED_PANELS_DIR = REPO_ROOT / "data" / "stitched_panels"

STRICT_ROLES = ("front", "nutrition", "price-tag")

# Tell the missing-input system that ``manifest.json`` files under
# ``stitched_panels/`` are produced by this script. Other steps that
# declare a manifest as input (extract_yaml.process_group_to_yaml)
# get a useful hint when the manifest hasn't been built yet.
register_producer(
    "data/stitched_panels/", "uv run python scripts/group_pipeline.py <gid>"
)


def _process_group_inputs(
    group: JsonObject, *, out_root: Path
) -> tuple[Path, ...]:
    """Resolve the durable input paths for one process_group call.

    Every photo referenced by the group must exist on disk before the
    pipeline runs. ``out_root`` is unused but takes the kwarg so the
    decorator's resolver signature lines up with the wrapped function.
    """
    del out_root
    photos = group.get("photos")
    if not isinstance(photos, list):
        return ()
    paths: list[Path] = []
    for ph in photos:
        if not isinstance(ph, dict):
            continue
        path_value = ph.get("path")
        if isinstance(path_value, str):
            paths.append(REPO_ROOT / path_value)
    return tuple(paths)


def _process_group_outputs(
    group: JsonObject, *, out_root: Path
) -> tuple[Path, ...]:
    """Resolve the durable output paths a process_group call will write.

    The per-photo crops are also outputs but we can't enumerate them
    until the bbox detector has run; declaring just the manifest is a
    correct (if coarse) summary, and the manifest itself records the
    crop paths.
    """
    gid_value = group.get("id")
    if not isinstance(gid_value, str):
        return ()
    return (out_root / gid_value / "manifest.json",)


@dataclass(frozen=True)
class GroupArtifacts:
    """Per-role outputs for a single group.

    Two views of the same crops:

    * ``panels_by_role`` — single image path per role. For multi-shot
      roles this is the stitched output; for single-shot it's the
      crop itself. Used by extractors that want one image (front,
      price-tag).
    * ``crops_by_role`` — full per-photo crop list per role, in
      capture order. Used by extractors that want to send each crop
      as a separate attachment to the model rather than working from
      a stitched/resampled composite (the nutrition multi-image path
      that avoids stitch artifacts).

    ``crop_sources`` maps each crop path to the source photo it was
    cropped from. Needed by review tooling that wants to display the
    crop alongside the original.
    """

    group_id: str
    store: str
    panels_by_role: dict[str, Path]
    crops_by_role: dict[str, tuple[Path, ...]]
    crop_sources: dict[Path, Path]
    # role → True if the panels_by_role path came from stitching
    # multiple crops; False if it's a single crop.
    stitched: dict[str, bool]


def assign_expected_roles(photo_roles: set[str]) -> tuple[str, ...]:
    """Restrict the photo's gold roles to known role kinds.

    Returns roles in :data:`role_bbox.ROLE_KINDS` order so the prompt
    enumeration is deterministic across runs.
    """
    return tuple(r for r in ROLE_KINDS if r in photo_roles)


@dataclass(frozen=True)
class CropResult:
    """One crop produced by detect-and-crop for a photo."""

    crop_path: Path
    source_photo: Path
    bbox: PanelBbox


def detect_and_crop_group(
    group: JsonObject,
    *,
    crop_dir: Path,
) -> dict[str, list[CropResult]]:
    """Run bbox detection + crop for every photo in a group.

    Returns role → list of crop results in capture order. Photos
    whose gold entry has no known-role kinds are skipped (they're
    loose-only and we don't have ground truth to constrain them).
    """
    by_role: dict[str, list[CropResult]] = defaultdict(list)
    photos = group.get("photos") or []
    if not isinstance(photos, list):
        return by_role
    for ph in photos:
        if not isinstance(ph, dict):
            continue
        path_value = ph.get("path")
        roles_value = ph.get("roles")
        if not isinstance(path_value, str):
            continue
        photo_path = REPO_ROOT / path_value
        roles: set[str] = (
            {r for r in roles_value if isinstance(r, str)}
            if isinstance(roles_value, list)
            else set()
        )
        expected = assign_expected_roles(roles)
        if not expected:
            print(
                f"  skipping {photo_path.name}: no known-role kinds in gold",
                file=sys.stderr,
                flush=True,
            )
            continue
        panels = detect_panels(photo_path, expected_roles=expected)
        if not panels:
            print(
                f"  WARN {photo_path.name}: detector returned no panels "
                f"for expected={expected}",
                file=sys.stderr,
                flush=True,
            )
            continue
        for panel in panels:
            cropped = crop_panel(photo_path, panel, out_dir=crop_dir)
            by_role[panel.kind].append(
                CropResult(
                    crop_path=cropped, source_photo=photo_path, bbox=panel
                )
            )
    return by_role


def stitch_role_outputs(
    group_id: str,
    by_role: dict[str, list[CropResult]],
    *,
    out_dir: Path,
) -> GroupArtifacts:
    """Per-role: single shot → use directly; multi → stitch on text axis.

    The single-shot bypass is the user's optimization: when a role has
    one crop, sizing+stitching adds no value (no other frame to align
    with), so we just keep the crop as-is.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_by_role: dict[str, Path] = {}
    crops_by_role: dict[str, tuple[Path, ...]] = {}
    crop_sources: dict[Path, Path] = {}
    stitched_flag: dict[str, bool] = {}
    for role, items in by_role.items():
        if not items:
            continue
        crop_paths = [item.crop_path for item in items]
        bboxes = [item.bbox for item in items]
        crops_by_role[role] = tuple(crop_paths)
        for item in items:
            crop_sources[item.crop_path] = item.source_photo
        if len(crop_paths) == 1:
            # Single-shot bypass: the crop IS the per-role output.
            panels_by_role[role] = crop_paths[0]
            stitched_flag[role] = False
        else:
            axis = stitch_axis_for_panels(bboxes)
            stitched_path = out_dir / f"{role}.jpg"
            stitch_role_panels(crop_paths, axis, stitched_path)
            panels_by_role[role] = stitched_path
            stitched_flag[role] = True

    return GroupArtifacts(
        group_id=group_id,
        store=_store_for_group_id(group_id),
        panels_by_role=panels_by_role,
        crops_by_role=crops_by_role,
        crop_sources=crop_sources,
        stitched=stitched_flag,
    )


def _store_for_group_id(group_id: str) -> str:
    """Derive the store from the deterministic group-id prefix."""
    if group_id.startswith("20260426_mom"):
        return "MOM"
    if group_id.startswith("20260426_cvs"):
        return "CVS"
    return "UNKNOWN"


def write_manifest(artifacts: GroupArtifacts, manifest_path: Path) -> None:
    """Write a JSON manifest of the canonical crops produced by this run.

    The per-group output directory accumulates stale crops as the bbox
    detector evolves (different prompts → different bbox coords →
    different filenames; old files don't get cleaned up). The manifest
    records *exactly* which crops belong to the latest run, so review
    tooling and future codegen can ignore the stragglers.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    crop_entries: list[dict[str, object]] = []
    for role, crops in artifacts.crops_by_role.items():
        for crop_path in crops:
            source = artifacts.crop_sources[crop_path]
            crop_entries.append(
                {
                    "role": role,
                    "crop": _repo_relative(crop_path),
                    "source_photo": _repo_relative(source),
                }
            )
    payload: dict[str, object] = {
        "group_id": artifacts.group_id,
        "store": artifacts.store,
        "crops": crop_entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))


def _repo_relative(path: Path) -> str:
    """Return ``path`` as a string relative to :data:`REPO_ROOT` when possible."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@pipeline_step(
    inputs=_process_group_inputs,
    outputs=_process_group_outputs,
    name="group_pipeline.process_group",
)
def process_group(group: JsonObject, *, out_root: Path) -> GroupArtifacts:
    """End-to-end: bbox + crop + per-role stitch for one group."""
    gid_value = group.get("id")
    if not isinstance(gid_value, str):
        raise ValueError(f"Group missing string 'id' field: {group}")
    gid = gid_value
    photos = group.get("photos")
    n_photos = len(photos) if isinstance(photos, list) else 0
    print(
        f"[pipeline] {gid}: {n_photos} photos",
        file=sys.stderr,
        flush=True,
    )
    group_dir = out_root / gid
    by_role = detect_and_crop_group(group, crop_dir=group_dir / "crops")
    artifacts = stitch_role_outputs(gid, by_role, out_dir=group_dir)
    write_manifest(artifacts, group_dir / "manifest.json")
    summary = ", ".join(
        f"{r}={'stitched' if artifacts.stitched[r] else 'single'}"
        for r in STRICT_ROLES
        if r in artifacts.panels_by_role
    )
    print(f"[pipeline] {gid}: {summary}", file=sys.stderr, flush=True)
    return artifacts


def load_group(gold_path: Path, group_id: str) -> JsonObject:
    """Look up one group dict by id from the gold-groups JSON file."""
    payload: JsonValue = json.loads(gold_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {gold_path}")
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"Expected 'groups' list in {gold_path}")
    for g in groups:
        if (
            isinstance(g, dict)
            and isinstance(g.get("id"), str)
            and g["id"] == group_id
        ):
            return g
    raise KeyError(f"Group {group_id!r} not in {gold_path}")


def main() -> int:  # pragma: no cover — CLI entry, exercised manually
    """Command-line entry: bbox + crop + stitch + manifest for each group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group_ids",
        nargs="+",
        help="One or more group IDs to process (e.g. 20260426_mom_001).",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=GOLD_GROUPS_JSON,
        help="Gold-groups JSON path.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=STITCHED_PANELS_DIR,
        help="Root directory for per-group stitched outputs.",
    )
    args = parser.parse_args()

    for gid in args.group_ids:
        group = load_group(args.gold, gid)
        process_group(group, out_root=args.out_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
