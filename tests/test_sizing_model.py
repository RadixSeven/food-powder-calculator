"""Tests for scripts/sizing_model.py.

The PyMC inference is exercised once on a tiny synthetic dataset (slow but
informative); decision functions are tested with hand-constructed posteriors
to keep the rest of the suite fast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sizing_model import (
    Posterior,
    Probe,
    chosen_size,
    expected_cost_of_more_probes,
    expected_savings_from_more_probes,
    fit_posterior,
    predictive_fail_probability,
    should_stop,
    tokens_for_size,
)


def _hand_posterior(
    mean_log_s: float,
    std_log_s: float,
    n_photos: int = 10,
    n_samples: int = 500,
) -> Posterior:
    """Approximation: posterior shape is ~degenerate at one (mu, sigma) pair."""
    return Posterior(
        mu_samples=np.full(n_samples, mean_log_s),
        sigma_samples=np.full(n_samples, std_log_s),
        n_photos_observed=n_photos,
    )


def test_predictive_fail_probability_monotonic_in_size() -> None:
    posterior = _hand_posterior(mean_log_s=math.log(512), std_log_s=0.3)
    p_small = predictive_fail_probability(posterior, 256)
    p_med = predictive_fail_probability(posterior, 512)
    p_large = predictive_fail_probability(posterior, 2048)
    assert p_small > p_med > p_large
    # At the population mean, P(s > X) should be ~0.5.
    assert 0.4 < p_med < 0.6


def test_predictive_fail_probability_zero_size_is_one() -> None:
    posterior = _hand_posterior(mean_log_s=math.log(512), std_log_s=0.3)
    assert predictive_fail_probability(posterior, 0) == 1.0


def test_chosen_size_meets_threshold() -> None:
    posterior = _hand_posterior(mean_log_s=math.log(512), std_log_s=0.3)
    n_remaining = 200
    threshold = 0.001
    x = chosen_size(posterior, n_remaining, cumulative_fail_threshold=threshold)
    per_photo_threshold = threshold / n_remaining
    assert predictive_fail_probability(posterior, x) < per_photo_threshold
    # And one px smaller would not meet the threshold.
    if x > 64:
        assert (
            predictive_fail_probability(posterior, x - 1) >= per_photo_threshold
        )


def test_chosen_size_grows_with_population_uncertainty() -> None:
    tight = _hand_posterior(mean_log_s=math.log(512), std_log_s=0.1)
    loose = _hand_posterior(mean_log_s=math.log(512), std_log_s=0.6)
    x_tight = chosen_size(tight, 100)
    x_loose = chosen_size(loose, 100)
    assert x_loose > x_tight


def test_chosen_size_rejects_zero_remaining() -> None:
    posterior = _hand_posterior(mean_log_s=math.log(512), std_log_s=0.3)
    with pytest.raises(ValueError, match="n_remaining_photos"):
        chosen_size(posterior, 0)


def test_tokens_for_size_quadratic() -> None:
    t1 = tokens_for_size(1024)
    t2 = tokens_for_size(2048)
    assert t2 == pytest.approx(4 * t1, rel=1e-6)


def test_tokens_for_size_zero() -> None:
    assert tokens_for_size(0) == 0.0


def test_expected_cost_grows_with_more_photos() -> None:
    cost_a = expected_cost_of_more_probes(
        n_more_photos=10,
        expected_probes_per_photo=4,
        probe_size_px=2048,
        cost_per_token=3e-6,
    )
    cost_b = expected_cost_of_more_probes(
        n_more_photos=20,
        expected_probes_per_photo=4,
        probe_size_px=2048,
        cost_per_token=3e-6,
    )
    assert cost_b == pytest.approx(2 * cost_a, rel=1e-6)


def test_expected_savings_zero_when_zero_inputs() -> None:
    posterior = _hand_posterior(mean_log_s=math.log(512), std_log_s=0.3)
    assert (
        expected_savings_from_more_probes(
            posterior_now=posterior,
            n_remaining_downstream_photos=100,
            n_more_probe_photos=0,
            cost_per_token=3e-6,
        )
        == 0.0
    )
    assert (
        expected_savings_from_more_probes(
            posterior_now=posterior,
            n_remaining_downstream_photos=0,
            n_more_probe_photos=10,
            cost_per_token=3e-6,
        )
        == 0.0
    )


def test_expected_savings_nonnegative_when_posterior_is_loose() -> None:
    """A loose posterior should benefit from more probes (savings > 0)."""
    posterior = _hand_posterior(
        mean_log_s=math.log(512), std_log_s=0.6, n_photos=10
    )
    savings = expected_savings_from_more_probes(
        posterior_now=posterior,
        n_remaining_downstream_photos=200,
        n_more_probe_photos=20,
        cost_per_token=3e-6,
    )
    assert savings >= 0.0


def test_should_stop_when_posterior_is_essentially_degenerate() -> None:
    """If sigma is already ~zero, narrowing it further can't reduce X*."""
    tight = _hand_posterior(
        mean_log_s=math.log(512), std_log_s=1e-6, n_photos=100
    )
    stop = should_stop(
        posterior_now=tight,
        n_remaining_downstream_photos=200,
        n_more_probe_photos=10,
        expected_probes_per_photo=4,
        probe_size_px=2048,
        cost_per_token=3e-6,
    )
    assert stop is True


def test_should_not_stop_when_posterior_is_loose_with_room_to_shrink() -> None:
    """A loose posterior with a chosen X below the upper bound benefits from
    narrowing, so the rule should keep going."""
    loose = _hand_posterior(
        mean_log_s=math.log(1024), std_log_s=0.4, n_photos=10
    )
    stop = should_stop(
        posterior_now=loose,
        n_remaining_downstream_photos=500,
        n_more_probe_photos=20,
        expected_probes_per_photo=4,
        probe_size_px=2048,
        cost_per_token=3e-6,
    )
    assert stop is False


def test_fit_posterior_recovers_population_signal() -> None:
    """End-to-end smoke test: feed the model probes generated from a known s_i
    and assert the posterior mean of mu lands near the truth."""
    rng = np.random.default_rng(42)
    true_log_s = rng.normal(loc=math.log(512), scale=0.3, size=8)
    probes: list[Probe] = []
    for i, log_s in enumerate(true_log_s):
        for size in (128, 256, 512, 1024, 2048):
            # Ground-truth: match if size > s_i, with no noise.
            matched = math.log(size) >= log_s
            probes.append(Probe(photo_id=f"p{i}", size=size, matched=matched))

    posterior = fit_posterior(probes, draws=200, tune=200, chains=2, seed=11)

    assert posterior.n_photos_observed == 8
    median_mu = float(np.median(posterior.mu_samples))
    sample_truth_mean = float(np.mean(true_log_s))
    # Print on assertion failure so we can see what the model returned.
    assert abs(median_mu - sample_truth_mean) < 0.6, (
        f"median_mu={median_mu:.3f}, sample_truth_mean={sample_truth_mean:.3f}, "
        f"diff={median_mu - sample_truth_mean:+.3f}"
    )


def test_fit_posterior_rejects_empty_probes() -> None:
    with pytest.raises(ValueError, match="at least one probe"):
        fit_posterior([], draws=10, tune=10, chains=1)
