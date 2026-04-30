"""Look up the run that produced a given path.

Walks ``data/runs/*/manifest.yaml`` and returns the most recent run
whose ``outputs`` includes the requested path. Useful when a YAML or
manifest file is unexpected or under suspicion: ``scripts/provenance.py
data/extracted_yaml/<gid>.yaml`` answers "which command produced this,
when, and from what?".

Output is one row per matched run, most-recent first (so the head of
the list is what you almost certainly want). Each row includes
``run_id``, ``git_sha``, ``argv``, the matched output's recorded SHA,
and the count of inputs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from _run import RUNS_DIR


@dataclass(frozen=True)
class ProvenanceMatch:
    """One run whose manifest lists the queried path among its outputs."""

    run_id: str
    git_sha: str
    git_dirty: bool
    started_at: str
    ended_at: str
    argv: tuple[str, ...]
    matched_path: str
    matched_sha256: str
    n_inputs: int
    step_name: str


def find_runs_producing(
    target: Path, *, runs_dir: Path = RUNS_DIR
) -> list[ProvenanceMatch]:
    """Return matches sorted most-recent-run-first.

    A "match" is any step in a manifest whose ``outputs`` includes
    ``target`` as either an absolute or repo-relative string. We
    accept either form because manifests historically captured
    absolute paths from the running machine; future runs may switch
    to repo-relative.
    """
    target_str = str(target)
    target_resolved = str(target.resolve()) if target.exists() else target_str
    matches: list[ProvenanceMatch] = []
    if not runs_dir.exists():
        return matches
    for manifest_path in runs_dir.glob("*/manifest.yaml"):
        match = _scan_manifest(
            manifest_path,
            target_str=target_str,
            target_resolved=target_resolved,
        )
        if match is not None:
            matches.append(match)
    # Sort by started_at descending (string ISO timestamps sort lexically).
    matches.sort(key=lambda m: m.started_at, reverse=True)
    return matches


def _scan_manifest(
    manifest_path: Path, *, target_str: str, target_resolved: str
) -> ProvenanceMatch | None:
    """Return a match for the manifest if any step lists the target."""
    payload = yaml.safe_load(manifest_path.read_text())
    if not isinstance(payload, dict):
        return None
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, dict):
            continue
        outputs = step.get("outputs")
        if not isinstance(outputs, list):
            continue
        for out in outputs:
            if not isinstance(out, dict):
                continue
            path_value = out.get("path")
            if path_value not in (target_str, target_resolved):
                continue
            sha = out.get("sha256")
            return ProvenanceMatch(
                run_id=cast(str, payload.get("run_id", "")),
                git_sha=cast(str, payload.get("git_sha", "")),
                git_dirty=bool(payload.get("git_dirty", False)),
                started_at=cast(str, payload.get("started_at", "")),
                ended_at=cast(str, payload.get("ended_at", "")),
                argv=tuple(cast(list[str], payload.get("argv") or [])),
                matched_path=cast(str, path_value),
                matched_sha256=sha if isinstance(sha, str) else "",
                n_inputs=len(step.get("inputs") or []),
                step_name=cast(str, step.get("name", "")),
            )
    return None


def format_matches(target: Path, matches: list[ProvenanceMatch]) -> str:
    """Render the provenance result for human reading.

    Most recent run first; the head of the list is almost certainly
    the answer you want.
    """
    if not matches:
        return f"No runs in data/runs/ produced {target}\n"
    lines = [f"Runs that produced {target} (most recent first):", ""]
    for m in matches:
        dirty = " (dirty)" if m.git_dirty else ""
        lines.append(f"  run_id:     {m.run_id}")
        lines.append(f"  git_sha:    {m.git_sha[:12]}{dirty}")
        lines.append(f"  step:       {m.step_name}")
        lines.append(f"  started:    {m.started_at}")
        lines.append(f"  ended:      {m.ended_at}")
        lines.append(f"  argv:       {' '.join(m.argv)}")
        lines.append(f"  output sha: {m.matched_sha256}")
        lines.append(f"  inputs:     {m.n_inputs} (see manifest for full list)")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Command-line entry: print provenance for one path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Path,
        help="Path to look up (absolute or relative).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Where to scan for run manifests (data/runs/ by default).",
    )
    args = parser.parse_args()
    matches = find_runs_producing(args.target, runs_dir=args.runs_dir)
    sys.stdout.write(format_matches(args.target, matches))
    return 0 if matches else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
