"""Run the per-photo binary search + Bayesian posterior across the photo
population and emit a chosen-size recommendation for the YAML-extraction
stage.

The flow:

1. Enumerate every PXL_*.jpg in ``data/raw_photos/`` (chronological order).
   Sizing is for downstream text extraction, so the population we care
   about is *all* photos that go into a stitched image — not just photos
   the grouping happened to label ``nutrition``.
2. For each candidate photo:

   a. ``select_search_model`` finds the cheapest model whose full-res
      extraction matches Opus 4.7's reference.
   b. ``binary_search_min_size`` runs the binary search (with the
      non-monotonic correction) and emits a list of ``Probe`` outcomes.
   c. After every photo (past the seed size) we refit :func:`fit_posterior`
      and compute :func:`chosen_size` for the remaining unseen photos, and
      call :func:`should_stop` to decide whether more probes are worth
      running.

3. Stop when the should_stop rule fires. There is no hard photo cap — if
   the posterior's uncertainty justifies it, every photo gets probed. The
   operator may abort with Ctrl-C.
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
from dataclasses import asdict, dataclass
from pathlib import Path

from find_legible_size import (
    binary_search_min_size,
    extract_payload,
    select_search_model,
)
from sizing_model import (
    Posterior,
    Probe,
    chosen_size,
    fit_posterior,
    should_stop,
)


@dataclass(frozen=True)
class PosteriorSummary:
    """Compact summary of a fit posterior for round-tripping to JSON."""

    mu_mean: float
    mu_std: float
    sigma_mean: float
    sigma_std: float
    n_photos_observed: int


@dataclass(frozen=True)
class PerPhotoResult:
    """One probed photo's outcome — what model was elected and what probes ran."""

    photo: str  # repo-relative
    search_model: str
    min_legible_size: int
    probes: list[Probe]


@dataclass(frozen=True)
class SizingResults:
    """Full output of :func:`run_sizing`. Serialized as ``data/sizing_results.json``."""

    chosen_size_px: int
    posterior_summary: PosteriorSummary
    n_candidate_photos: int
    per_photo: list[PerPhotoResult]


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PHOTOS_DIR = REPO_ROOT / "data" / "raw_photos"
OUTPUT_JSON = REPO_ROOT / "data" / "sizing_results.json"

# Used to convert "expected savings" to the same units as
# expected_cost_of_more_probes. Sonnet's per-input-token rate as of 2026-04
# (~$3/M input). Both numbers feed the should_stop rule, so the absolute
# value matters less than that they're in the same units.
COST_PER_TOKEN_USD = 3e-6


def list_candidate_photos(photo_dir: Path) -> list[Path]:
    """Every PXL_*.jpg in ``photo_dir``, in chronological order.

    All raw photos are candidates because every photo eventually goes
    through the YAML-extraction stitched image; the sizing decision has to
    be safe for the whole population, not just photos labeled ``nutrition``
    by the grouping. The Bayesian posterior naturally weights difficult
    (high-resolution-required) photos when fitting.
    """
    return sorted(photo_dir.glob("PXL_*.jpg"))


def run_sizing(
    candidate_photos: list[Path],
    *,
    seed_size: int = 10,
) -> SizingResults:
    """Drive the per-photo loop and return the final results.

    ``seed_size``: probe this many photos before the stop rule starts firing
    (the rule relies on a posterior, which needs ~10 observations — about
    2-3 product groups — before it carries useful signal).

    The loop runs until the cost-benefit stop rule fires. There is
    deliberately no max-photos cap: if the population's per-photo
    uncertainty justifies it, every candidate gets probed. The operator can
    Ctrl-C if they want to abort.
    """
    probes: list[Probe] = []
    per_photo: list[PerPhotoResult] = []

    for i, photo in enumerate(candidate_photos):
        print(
            f"[sizing] photo {i + 1}/{len(candidate_photos)}: {photo.name}",
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
            PerPhotoResult(
                photo=str(photo.relative_to(REPO_ROOT)),
                search_model=search_model,
                min_legible_size=result.min_legible_size,
                probes=list(result.probes),
            )
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

    final_posterior = fit_posterior(probes)
    n_remaining = max(1, len(candidate_photos) - len(per_photo))
    return SizingResults(
        chosen_size_px=chosen_size(final_posterior, n_remaining),
        posterior_summary=_summarize_posterior(final_posterior),
        n_candidate_photos=len(candidate_photos),
        per_photo=per_photo,
    )


def _summarize_posterior(posterior: Posterior) -> PosteriorSummary:
    return PosteriorSummary(
        mu_mean=float(posterior.mu_samples.mean()),
        mu_std=float(posterior.mu_samples.std()),
        sigma_mean=float(posterior.sigma_samples.mean()),
        sigma_std=float(posterior.sigma_samples.std()),
        n_photos_observed=posterior.n_photos_observed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the sizing study and write data/sizing_results.json."
    )
    parser.add_argument(
        "--photo-dir",
        type=Path,
        default=RAW_PHOTOS_DIR,
        help="Source directory of PXL_*.jpg photos.",
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
        default=10,
        help=(
            "Probes this many photos before the stop rule can fire. ~10 is "
            "two to three product groups — barely enough to anchor the "
            "posterior."
        ),
    )
    args = parser.parse_args()

    candidates = list_candidate_photos(args.photo_dir)
    print(
        f"[sizing] {len(candidates)} candidate photos; the cost-stop rule "
        "will decide where to stop",
        file=sys.stderr,
        flush=True,
    )
    results = run_sizing(candidates, seed_size=args.seed_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(results), indent=2))
    print(
        f"[sizing] DONE — chosen_size={results.chosen_size_px}px, "
        f"wrote {args.out}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
