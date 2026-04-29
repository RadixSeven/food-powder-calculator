"""Bayesian sizing model for the per-photo legibility problem.

Given a set of probes (per-photo (size, matched) outcomes), fit a population
log-normal over per-photo minimum legible sizes and provide:

* :func:`chosen_size` — smallest longest-side that keeps the cumulative
  failure probability across remaining photos below a threshold.
* :func:`expected_savings_from_more_probes` — token-equivalent savings
  expected if we run binary searches on more photos.
* :func:`expected_cost_of_more_probes` — token-equivalent cost of those
  searches.
* :func:`should_stop` — True if the cost almost certainly exceeds the
  savings (P > 0.995).

The PyMC model is wrapped in :func:`fit_posterior` and returns a small
typed :class:`Posterior` so the decision functions can be unit-tested
without sampling.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# PyTensor (PyMC's compute backend) tries to compile C extensions on import.
# In pants' pex test sandbox we don't have Python.h available, so flip to the
# Python-only backend before importing pymc. Inference is slower but works
# without system headers, which is what tests need.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=,mode=FAST_COMPILE")

import numpy as np
import pymc as pm
from numpy.typing import NDArray

# Ramp width on the log-size axis for the probe likelihood. Sets how
# sharply ``P(match)`` transitions from 0 to 1 as ``log size`` crosses
# ``log s_i``. The 95%-band spans ``±1.65 · tau`` log units (= a factor of
# e^(3.3·tau) in pixels). 0.30 gives a transition width of ~factor-of-2.7
# in pixels — informative but soft enough to absorb the noise that lucky-
# pass probes inject during binary search.
LOG_SIZE_RAMP_TAU = 0.30

# Floor / ceiling on the per-probe Bernoulli probability. Without this,
# the prior + jittered starting point can put a photo's ``log s_i`` far
# from a probe size, sending ``log P`` to -inf and crashing NUTS at
# initialization.
P_MATCH_FLOOR = 1e-6


@dataclass(frozen=True)
class Probe:
    """One binary-search probe outcome."""

    photo_id: str
    size: int  # longest-side px at which the search model was evaluated
    matched: bool  # did the search model's extraction match the reference?


@dataclass
class Posterior:
    """A small wrapper around posterior samples needed for the decision rules.

    All log-scale quantities are natural log of pixels.
    """

    mu_samples: NDArray[np.float64]  # population mean of log(s_i)
    sigma_samples: NDArray[np.float64]  # population std of log(s_i)
    n_photos_observed: int  # number of distinct photos that contributed probes


PRIOR_MU = math.log(980)
PRIOR_SIGMA = 1.076032 * math.log(2)


def fit_posterior(
    probes: list[Probe],
    *,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    seed: int = 7,
) -> Posterior:
    """Fit a Bayesian model to the probe outcomes and return posterior samples.

    The model:

    * ``log s_i ~ Normal(mu, sigma)`` per photo.
    * ``mu ~ Normal(log 980, 1.076032 * log 2)``  — feasibility-bounded prior.
      Lower bound: longest-side 320 px (the user reports being able to read a
      nutrition label at 241×320, so 320 is a plausible floor for the
      population's smallest legible size). Upper bound: 2998 px (well below
      the camera's max dimension of 4080 — anything close to that is
      effectively the original photo). On a base-2 log scale the bounds are
      lg 320 ≈ 8.32 and lg 2998 ≈ 11.55, giving lg-mean ≈ 9.937 (= lg 980).
      Code expresses both in natural-log units.
    * ``sigma ~ HalfNormal(0.5)``  — order-of-magnitude variation tolerated.
    * ``P(match | x, s_i) = Phi((log x - log s_i) / tau)``.
    """
    if not probes:
        raise ValueError("fit_posterior requires at least one probe")

    photo_ids = sorted({p.photo_id for p in probes})
    photo_idx = {pid: i for i, pid in enumerate(photo_ids)}
    n_photos = len(photo_ids)

    log_sizes = np.array([math.log(p.size) for p in probes], dtype=np.float64)
    matched = np.array([1.0 if p.matched else 0.0 for p in probes])
    photo_for_probe = np.array([photo_idx[p.photo_id] for p in probes])

    with pm.Model():  # pyrefly: ignore[bad-context-manager]
        mu = pm.Normal("mu", mu=PRIOR_MU, sigma=PRIOR_SIGMA)
        sigma = pm.HalfNormal("sigma", sigma=0.5)
        log_s = pm.Normal("log_s", mu=mu, sigma=sigma, shape=n_photos)
        z = (log_sizes - log_s[photo_for_probe]) / LOG_SIZE_RAMP_TAU
        p_match = pm.math.clip(
            pm.math.invprobit(z), P_MATCH_FLOOR, 1.0 - P_MATCH_FLOOR
        )
        pm.Bernoulli("obs", p=p_match, observed=matched)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )

    posterior = idata.posterior  # pyrefly: ignore[missing-attribute]
    return Posterior(
        mu_samples=np.asarray(posterior["mu"].values).reshape(-1),
        sigma_samples=np.asarray(posterior["sigma"].values).reshape(-1),
        n_photos_observed=n_photos,
    )


def predictive_fail_probability(
    posterior: Posterior, longest_side_px: int
) -> float:
    """P(an unseen photo's min legible size > longest_side_px).

    Marginalizes over the joint posterior of (mu, sigma) by averaging the
    per-sample log-normal survival function.
    """
    if longest_side_px < 1:
        return 1.0
    log_x = math.log(longest_side_px)
    # Per posterior sample: 1 - Phi((log_x - mu) / sigma)
    z = (log_x - posterior.mu_samples) / np.maximum(
        posterior.sigma_samples, 1e-9
    )
    survivals = 0.5 * (1.0 - _erf(z / math.sqrt(2)))
    return float(np.mean(survivals))


def chosen_size(
    posterior: Posterior,
    n_remaining_photos: int,
    *,
    cumulative_fail_threshold: float = 0.001,
    upper_bound_px: int = 8192,
    lower_bound_px: int = 64,
) -> int:
    """Smallest longest-side at which expected failures across remaining photos < threshold.

    ``cumulative_fail_threshold`` is the maximum acceptable expected number of
    illegible reductions across the remaining photos: with the user-stated
    0.001 and ~200 remaining photos, expected illegibility count stays well
    below 1.
    """
    if n_remaining_photos < 1:
        raise ValueError("n_remaining_photos must be >= 1")
    per_photo_threshold = cumulative_fail_threshold / n_remaining_photos
    # Binary search over X: monotonically decreasing fail probability in X.
    lo, hi = lower_bound_px, upper_bound_px
    while lo < hi:
        mid = (lo + hi) // 2
        if predictive_fail_probability(posterior, mid) < per_photo_threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo


def tokens_for_size(longest_side_px: int, aspect_ratio: float = 0.75) -> float:
    """Approximate Anthropic vision tokens for an image at this longest side.

    Anthropic's rough rule: tokens ≈ width × height / 750. We use a default
    portrait-ish aspect ratio so the formula reduces to ≈ longest_side² × ratio / 750.
    """
    if longest_side_px < 1:
        return 0.0
    return (longest_side_px * longest_side_px * aspect_ratio) / 750.0


def expected_cost_of_more_probes(
    *,
    n_more_photos: int,
    expected_probes_per_photo: float,
    probe_size_px: int,
    cost_per_token: float,
) -> float:
    """Estimate cost (USD-equivalent) of running binary searches on more photos."""
    tokens_per_probe = tokens_for_size(probe_size_px)
    return (
        n_more_photos
        * expected_probes_per_photo
        * tokens_per_probe
        * cost_per_token
    )


def expected_savings_from_more_probes(
    *,
    posterior_now: Posterior,
    n_remaining_downstream_photos: int,
    n_more_probe_photos: int,
    downstream_calls_per_photo: int = 2,
    cost_per_token: float,
    cumulative_fail_threshold: float = 0.001,
) -> float:
    """Estimate token-cost savings of adding probes for ``n_more_probe_photos`` photos.

    Approximation: assume the posterior over ``mu``/``sigma`` shrinks
    proportionally to ``sqrt(N/(N+K))`` after adding K more photos. A
    narrower posterior leads to a smaller chosen_size for the same threshold,
    which compounds across the remaining downstream photos and calls.
    """
    if n_more_probe_photos < 1 or n_remaining_downstream_photos < 1:
        return 0.0

    n_now = posterior_now.n_photos_observed
    shrink_factor = math.sqrt(n_now / (n_now + n_more_probe_photos))

    narrowed = Posterior(
        mu_samples=posterior_now.mu_samples,
        sigma_samples=posterior_now.sigma_samples * shrink_factor,
        n_photos_observed=n_now + n_more_probe_photos,
    )

    x_now = chosen_size(
        posterior_now,
        n_remaining_downstream_photos,
        cumulative_fail_threshold=cumulative_fail_threshold,
    )
    x_narrowed = chosen_size(
        narrowed,
        n_remaining_downstream_photos,
        cumulative_fail_threshold=cumulative_fail_threshold,
    )
    if x_narrowed >= x_now:
        return 0.0
    tokens_saved_per_call = tokens_for_size(x_now) - tokens_for_size(x_narrowed)
    return (
        tokens_saved_per_call
        * downstream_calls_per_photo
        * n_remaining_downstream_photos
        * cost_per_token
    )


def should_stop(
    *,
    posterior_now: Posterior,
    n_remaining_downstream_photos: int,
    n_more_probe_photos: int,
    expected_probes_per_photo: float,
    probe_size_px: int,
    cost_per_token: float,
    confidence: float = 0.995,
) -> bool:
    """Return True if more probes are very unlikely to pay back.

    We're not Monte-Carloing the cost/savings posterior here — both quantities
    are approximations. The user-specified rule "stop when P(cost > savings)
    > 0.995" maps to: stop when (expected cost) * confidence_buffer >
    (expected savings). The confidence parameter controls the buffer.
    """
    cost = expected_cost_of_more_probes(
        n_more_photos=n_more_probe_photos,
        expected_probes_per_photo=expected_probes_per_photo,
        probe_size_px=probe_size_px,
        cost_per_token=cost_per_token,
    )
    savings = expected_savings_from_more_probes(
        posterior_now=posterior_now,
        n_remaining_downstream_photos=n_remaining_downstream_photos,
        n_more_probe_photos=n_more_probe_photos,
        cost_per_token=cost_per_token,
    )
    # If we're 99.5% confident savings won't exceed cost, stop. Approximate
    # this by requiring the expected savings to be a (1 - confidence)-fraction
    # of the expected cost or smaller.
    return savings <= cost * (1.0 - confidence)


def _erf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized erf via numpy (avoid math.erf inside ndarray ops)."""
    # numpy 2.x doesn't expose np.erf; use scipy if available, else
    # Abramowitz & Stegun approximation.
    try:
        from scipy.special import erf as _scipy_erf  # noqa: PLC0415

        return np.asarray(_scipy_erf(x), dtype=np.float64)
    except ImportError:  # pragma: no cover — scipy is a transitive dep
        return _erf_approx(x)


def _erf_approx(
    x: NDArray[np.float64],
) -> NDArray[np.float64]:  # pragma: no cover
    """Abramowitz & Stegun 7.1.26 — fallback if scipy isn't installed."""
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(
        -ax * ax
    )
    return sign * y
