"""Run the per-photo binary search + Bayesian posterior across a sample of
nutrition-label photos and emit a chosen-size recommendation.

The flow:

1. Load ``data/gold_groups.json`` and pick photos whose roles include
   ``nutrition`` — those are the text-dense photos that actually drive the
   sizing decision. Fronts, price tags, etc. would only depress the
   population mean.
2. For each candidate photo, in chronological order:

   a. ``select_search_model`` finds the cheapest model whose full-res
      extraction matches Opus 4.7's reference.
   b. ``binary_search_min_size`` runs the binary search (with the
      non-monotonic correction) and emits a list of ``Probe`` outcomes.
   c. After every photo we refit :func:`fit_posterior` and compute
      :func:`chosen_size` for the *remaining* unseen photos and call
      :func:`should_stop` to decide whether more probes are worth running.

3. Stop when either the should_stop rule fires or the operator-supplied
   ``--max-photos`` cap is hit.
4. Write ``data/sizing_results.json`` with the final chosen size, posterior
   summary, and every probe — so the decision is reproducible.

Costs per photo (rough): one Opus full-res reference (~$0.27) + one
Haiku/Sonnet match check (~$0.02) + ~6 binary-search probes at the chosen
search model. Total $0.30-0.40/photo. The Bayesian stop rule keeps the
total budget bounded.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from find_legible_size import (
    binary_search_min_size,
    extract_payload,
    select_search_model,
)
from sizing_model import (
    Probe,
    chosen_size,
    fit_posterior,
    should_stop,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLD_GROUPS_JSON = REPO_ROOT / "data" / "gold_groups.json"
OUTPUT_JSON = REPO_ROOT / "data" / "sizing_results.json"

# Used to convert "expected savings" to the same units as
# expected_cost_of_more_probes. Sonnet's per-input-token rate as of 2026-04
# (~$3/M input). Both numbers feed the should_stop rule, so the absolute
# value matters less than that they're in the same units.
COST_PER_TOKEN_USD = 3e-6


def list_nutrition_photos(gold_groups_json: Path) -> list[Path]:
    """Photos whose roles include ``nutrition``, in chronological order."""
    payload = json.loads(gold_groups_json.read_text())
    out: list[Path] = []
    for group in payload["groups"]:
        for p in group["photos"]:
            if "nutrition" in p["roles"]:
                out.append(REPO_ROOT / p["path"])
    return sorted(out)


def run_sizing(
    candidate_photos: list[Path],
    *,
    seed_size: int = 5,
    max_photos: int = 30,
) -> dict[str, object]:
    """Drive the per-photo loop and return the final results payload.

    ``seed_size``: probe this many photos before the stop rule starts firing
    (the rule relies on a posterior, which needs at least a few observations
    to be meaningful).
    """
    probes: list[Probe] = []
    per_photo: list[dict[str, object]] = []

    for i, photo in enumerate(candidate_photos):
        if i >= max_photos:
            break
        print(
            f"[sizing] photo {i + 1}/{min(max_photos, len(candidate_photos))}: "
            f"{photo.name}",
            file=sys.stderr,
            flush=True,
        )
        reference = extract_payload(photo, "opus")
        search_model, _ = select_search_model(photo, reference)
        result = binary_search_min_size(
            photo, search_model, reference, photo_id=photo.name
        )
        probes.extend(result.probes)
        per_photo.append(
            {
                "photo": str(photo.relative_to(REPO_ROOT)),
                "search_model": search_model,
                "min_legible_size": result.min_legible_size,
                "probes": [asdict(p) for p in result.probes],
            }
        )
        n_seen = i + 1
        if n_seen < seed_size:
            continue
        posterior = fit_posterior(probes)
        n_remaining = max(1, len(candidate_photos) - n_seen)
        x_chosen = chosen_size(posterior, n_remaining)
        print(
            f"[sizing] after {n_seen} photos: chosen_size={x_chosen}px "
            f"(posterior mu={float(posterior.mu_samples.mean()):.3f}, "
            f"sigma={float(posterior.sigma_samples.mean()):.3f})",
            file=sys.stderr,
            flush=True,
        )
        if should_stop(
            posterior_now=posterior,
            n_remaining_downstream_photos=n_remaining,
            n_more_probe_photos=1,
            expected_probes_per_photo=8,
            probe_size_px=x_chosen,
            cost_per_token=COST_PER_TOKEN_USD,
        ):
            print(
                f"[sizing] should_stop fired after {n_seen} photos",
                file=sys.stderr,
                flush=True,
            )
            break

    posterior = fit_posterior(probes)
    n_remaining = max(1, len(candidate_photos) - len(per_photo))
    x_chosen = chosen_size(posterior, n_remaining)
    return {
        "chosen_size_px": x_chosen,
        "posterior_summary": {
            "mu_mean": float(posterior.mu_samples.mean()),
            "mu_std": float(posterior.mu_samples.std()),
            "sigma_mean": float(posterior.sigma_samples.mean()),
            "sigma_std": float(posterior.sigma_samples.std()),
            "n_photos_observed": posterior.n_photos_observed,
        },
        "n_candidate_photos": len(candidate_photos),
        "per_photo": per_photo,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the sizing study and write data/sizing_results.json."
    )
    parser.add_argument(
        "--gold-groups",
        type=Path,
        default=GOLD_GROUPS_JSON,
        help="Source of nutrition photos.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_JSON,
        help="Destination JSON for the results.",
    )
    parser.add_argument(
        "--seed-size",
        type=int,
        default=5,
        help="Number of photos probed before the stop rule starts firing.",
    )
    parser.add_argument(
        "--max-photos",
        type=int,
        default=30,
        help="Hard upper bound on probed photos.",
    )
    args = parser.parse_args()

    candidates = list_nutrition_photos(args.gold_groups)
    print(
        f"[sizing] {len(candidates)} candidate nutrition photos; "
        f"will probe up to {args.max_photos}",
        file=sys.stderr,
        flush=True,
    )
    results = run_sizing(
        candidates,
        seed_size=args.seed_size,
        max_photos=args.max_photos,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(
        f"[sizing] DONE — chosen_size={results['chosen_size_px']}px, "
        f"wrote {args.out}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
