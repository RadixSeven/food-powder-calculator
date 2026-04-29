"""One-line-per-group summary of extracted YAML.

Surfaces in one place the things a reviewer wants to see at a glance:
which roles each group has, the front product name (sanity check vs
what the photos show), the price-tag UPC (sanity check the cross-
store extraction), and the nutrition row count (low row counts mean
the model gave up early).

Usage:

    uv run python scripts/yaml_summary.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTED_YAML_DIR = REPO_ROOT / "data" / "extracted_yaml"
GOLD_GROUPS_JSON = REPO_ROOT / "data" / "gold_groups.json"


@dataclass(frozen=True)
class Summary:
    """One row of the per-group review table — fields a human scans."""

    group_id: str
    store: str
    front_name: str
    front_mfr: str
    nutrition_rows: int
    nutrition_unknown_cells: int
    nutrition_total_cells: int
    price_cents: int | None
    price_upc: str
    price_size: str


def load_summary(yaml_path: Path) -> Summary:
    """Read a per-group YAML and project it down to a flat :class:`Summary`."""
    parsed = cast(dict[str, object], yaml.safe_load(yaml_path.read_text()))
    front = cast(dict[str, str], parsed.get("front") or {})
    nutrition = cast(dict[str, object], parsed.get("nutrition") or {})
    price = cast(dict[str, object], parsed.get("price_tag") or {})
    rows = cast(list[list[str]], nutrition.get("rows") or [])
    unknown = sum(1 for row in rows for c in row if c == "▮")
    total = sum(len(row) for row in rows)
    return Summary(
        group_id=cast(str, parsed.get("group_id", yaml_path.stem)),
        store=cast(str, parsed.get("store", "")),
        front_name=front.get("product_name", ""),
        front_mfr=front.get("manufacturer", ""),
        nutrition_rows=len(rows),
        nutrition_unknown_cells=unknown,
        nutrition_total_cells=total,
        price_cents=cast(int | None, price.get("price_cents")),
        price_upc=cast(str, price.get("upc", "")),
        price_size=cast(str, price.get("size", "")),
    )


def gold_group_count(gold_path: Path) -> int:
    """Count the groups in the gold JSON (denominator for the summary report)."""
    payload = cast(dict[str, object], json.loads(gold_path.read_text()))
    groups = cast(list[object], payload.get("groups") or [])
    return len(groups)


def report(summaries: list[Summary], expected_total: int) -> None:
    """Print the per-group summary table plus an aggregate footer."""
    print(
        f"{'group':22s} {'store':4s} "
        f"{'front':45s} {'rows':4s} {'unk%':5s} {'price':6s} "
        f"{'upc':14s} {'size':10s}"
    )
    print("-" * 130)
    for s in sorted(summaries, key=lambda x: x.group_id):
        unk_pct = (
            s.nutrition_unknown_cells / s.nutrition_total_cells
            if s.nutrition_total_cells
            else 0.0
        )
        price_str = (
            f"${s.price_cents / 100:5.2f}"
            if s.price_cents is not None
            else "  -  "
        )
        front = (s.front_name or "(no front)")[:43]
        upc = s.price_upc or "-"
        size = s.price_size or "-"
        print(
            f"{s.group_id:22s} {s.store:4s} "
            f"{front:45s} {s.nutrition_rows:4d} {unk_pct:5.1%} {price_str:6s} "
            f"{upc:14s} {size:10s}"
        )

    print()
    print(f"Extracted: {len(summaries)} / {expected_total} groups")
    n_complete = sum(
        1
        for s in summaries
        if s.front_name and s.nutrition_rows > 0 and s.price_cents is not None
    )
    n_no_price = sum(
        1
        for s in summaries
        if s.front_name and s.nutrition_rows > 0 and s.price_cents is None
    )
    n_other = len(summaries) - n_complete - n_no_price
    print(f"  fully populated (front+nutrition+price): {n_complete}")
    print(f"  front+nutrition only (no shelf-tag photo): {n_no_price}")
    print(f"  other (missing role): {n_other}")
    upc_lengths: dict[int, int] = {}
    for s in summaries:
        if s.price_upc:
            upc_lengths[len(s.price_upc)] = (
                upc_lengths.get(len(s.price_upc), 0) + 1
            )
    if upc_lengths:
        print("\nUPC field length distribution (length: count):")
        for length, count in sorted(upc_lengths.items()):
            tag = (
                "MOM 12-digit"
                if length == 12
                else "CVS 5-digit slug"
                if length == 5
                else ""
            )
            print(f"  {length:2d}: {count} {tag}")


def main() -> int:  # pragma: no cover — CLI entry, exercised manually
    """Command-line entry: load every YAML in a directory and print a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=EXTRACTED_YAML_DIR,
        help="Directory containing extracted YAMLs.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=GOLD_GROUPS_JSON,
        help="Gold-groups JSON (for total-count comparison).",
    )
    args = parser.parse_args()
    if not args.yaml_dir.exists():
        print(f"No YAML dir at {args.yaml_dir}", file=sys.stderr)
        return 1
    summaries = [load_summary(p) for p in sorted(args.yaml_dir.glob("*.yaml"))]
    expected = (
        gold_group_count(args.gold) if args.gold.exists() else len(summaries)
    )
    report(summaries, expected)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
