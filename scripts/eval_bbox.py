"""Evaluate role-bbox detection against the gold-groups labels.

Strict roles ({front, nutrition, price-tag}) are scored both ways:

* **Positive**: gold has role R → detector should detect R.
* **Negative**: gold lacks role R → detector should NOT detect R.

Loose roles ({ingredients, other-label}) are excluded from scoring per
the manual-verification scope of ``data/gold_groups.json`` — only
strict roles were exhaustively reviewed, so loose-role absence isn't a
reliable negative signal.

Usage:

    uv run python scripts/eval_bbox.py [--limit N]

Re-running with a modified BBOX_PROMPT busts the response cache via
the prompt-SHA key, so prompt iteration is just edit-and-rerun.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from role_bbox import detect_panels

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_GROUPS_JSON = REPO_ROOT / "data" / "gold_groups.json"

STRICT_ROLES = ("front", "nutrition", "price-tag")


@dataclass(frozen=True)
class PhotoEval:
    """One photo's gold vs detected roles, restricted to the strict set."""

    name: str
    gold_strict: frozenset[str]
    detected_strict: frozenset[str]
    detected_extras: frozenset[str]


def load_gold(path: Path) -> dict[str, frozenset[str]]:
    """Map gold photo path → all gold roles for that photo."""
    payload = json.loads(path.read_text())
    out: dict[str, frozenset[str]] = {}
    for g in payload["groups"]:
        for ph in g.get("photos", []):
            out[ph["path"]] = frozenset(ph.get("roles", []))
    return out


def evaluate(
    photos: list[Path], gold: dict[str, frozenset[str]]
) -> list[PhotoEval]:
    """Run detection on each photo and pair against gold strict roles.

    Calls the detector with ``expected_roles=STRICT_ROLES`` rather than
    declaring per-photo position-derived roles — this evaluation is
    measuring the detector's intrinsic role-classification accuracy
    among the strict set, not the position-rule pipeline.
    """
    out: list[PhotoEval] = []
    for p in photos:
        rel = str(p.relative_to(REPO_ROOT))
        gold_all = gold.get(rel, frozenset())
        gold_strict = gold_all & frozenset(STRICT_ROLES)
        panels = detect_panels(p, expected_roles=STRICT_ROLES)
        detected_all = frozenset(x.kind for x in panels)
        detected_strict = detected_all & frozenset(STRICT_ROLES)
        detected_extras = detected_all - frozenset(STRICT_ROLES)
        out.append(
            PhotoEval(
                name=p.name,
                gold_strict=gold_strict,
                detected_strict=detected_strict,
                detected_extras=detected_extras,
            )
        )
    return out


def report(evals: list[PhotoEval]) -> None:
    """Print per-role TP/FP/FN/TN counts plus the per-photo verdict table."""
    print(
        f"{'photo':45s} {'gold':28s} {'detected':28s} {'extras':25s} {'verdict':s}"
    )
    print("-" * 140)
    # Per-role: tp, fp, fn, tn
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    photo_perfect = 0
    for e in evals:
        verdicts: list[str] = []
        for r in STRICT_ROLES:
            in_gold = r in e.gold_strict
            in_det = r in e.detected_strict
            if in_gold and in_det:
                counts[r][0] += 1
            elif in_det and not in_gold:
                counts[r][1] += 1
                verdicts.append(f"FP:{r}")
            elif in_gold and not in_det:
                counts[r][2] += 1
                verdicts.append(f"FN:{r}")
            else:
                counts[r][3] += 1
        if not verdicts:
            photo_perfect += 1
            verdict_str = "OK"
        else:
            verdict_str = " ".join(verdicts)
        print(
            f"{e.name:45s} {str(sorted(e.gold_strict)):28s} "
            f"{str(sorted(e.detected_strict)):28s} "
            f"{str(sorted(e.detected_extras)):25s} {verdict_str}"
        )

    total_tp = sum(c[0] for c in counts.values())
    total_fp = sum(c[1] for c in counts.values())
    total_fn = sum(c[2] for c in counts.values())
    total_tn = sum(c[3] for c in counts.values())

    print()
    print(f"Photos: {len(evals)} ({photo_perfect} perfect)")
    print("Per-role (TP / FP / FN / TN):")
    for r in STRICT_ROLES:
        tp, fp, fn, tn = counts[r]
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        print(
            f"  {r:12s}  {tp:3d} / {fp:3d} / {fn:3d} / {tn:3d}   "
            f"P={prec:6.1%}  R={rec:6.1%}  Spec={spec:6.1%}"
        )

    overall_prec = total_tp / max(1, total_tp + total_fp)
    overall_rec = total_tp / max(1, total_tp + total_fn)
    overall_spec = total_tn / max(1, total_tn + total_fp)
    print(
        f"  {'OVERALL':12s}  {total_tp:3d} / {total_fp:3d} / "
        f"{total_fn:3d} / {total_tn:3d}   "
        f"P={overall_prec:6.1%}  R={overall_rec:6.1%}  "
        f"Spec={overall_spec:6.1%}"
    )


def main() -> int:
    """Command-line entry: load gold, run detection, print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N photos (sorted by filename).",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=GOLD_GROUPS_JSON,
        help="Gold-groups JSON path.",
    )
    args = parser.parse_args()

    gold = load_gold(args.gold)
    photos = sorted(REPO_ROOT / p for p in gold)
    if args.limit:
        photos = photos[: args.limit]
    print(f"Evaluating {len(photos)} photos against gold...")
    evals = evaluate(photos, gold)
    report(evals)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
