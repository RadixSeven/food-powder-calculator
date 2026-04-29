"""Tests for scripts/run_sizing.py.

Mocks the find_legible_size + sizing_model entry points so the orchestration
logic can be exercised without LLM calls or PyMC sampling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from find_legible_size import ExtractedPayload, SearchResult
from PIL import Image
from run_sizing import (
    PerPhotoResult,
    PosteriorSummary,
    SizingResults,
    list_candidate_photos,
    main,
    run_sizing,
)
from sizing_model import Posterior, Probe


@dataclass
class _FakePhotoSet:
    photo_dir: Path
    photos: list[Path]


@pytest.fixture
def fake_photos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _FakePhotoSet:
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    photos: list[Path] = []
    for name in (
        "PXL_20260426_001.jpg",
        "PXL_20260426_002.jpg",
        "PXL_20260426_003.jpg",
    ):
        p = photo_dir / name
        Image.new("RGB", (8, 8), (200, 50, 50)).save(p, "JPEG")
        photos.append(p)
    # Stray non-PXL file should be ignored by list_candidate_photos.
    (photo_dir / "ignore_me.txt").write_text("nope")
    monkeypatch.setattr("run_sizing.REPO_ROOT", tmp_path)
    return _FakePhotoSet(photo_dir=photo_dir, photos=photos)


def test_list_candidate_photos_returns_pxl_files_in_order(
    fake_photos: _FakePhotoSet,
) -> None:
    found = list_candidate_photos(fake_photos.photo_dir)
    assert [p.name for p in found] == [
        "PXL_20260426_001.jpg",
        "PXL_20260426_002.jpg",
        "PXL_20260426_003.jpg",
    ]


def _hand_posterior(
    mu_mean: float = 6.5, sigma_mean: float = 0.3, n_photos: int = 5
) -> Posterior:
    return Posterior(
        mu_samples=np.full(200, mu_mean),
        sigma_samples=np.full(200, sigma_mean),
        n_photos_observed=n_photos,
    )


def _wire_mocks(
    monkeypatch: pytest.MonkeyPatch, *, should_stop: bool
) -> MagicMock:
    """Stub out the LLM-bound entry points; return the fit_posterior mock."""

    def fake_search(*_a: object, **k: object) -> SearchResult:
        pid = cast(str, k["photo_id"])
        return SearchResult(
            photo_id=pid,
            reference_model="opus",
            search_model="haiku",
            min_legible_size=512,
            probes=[Probe(photo_id=pid, size=512, matched=True)],
        )

    monkeypatch.setattr(
        "run_sizing.extract_payload",
        lambda *_a, **_k: ExtractedPayload(text="hi", barcodes=()),
    )
    monkeypatch.setattr(
        "run_sizing.select_search_model",
        lambda *_a, **_k: ("haiku", ExtractedPayload(text="hi", barcodes=())),
    )
    monkeypatch.setattr("run_sizing.binary_search_min_size", fake_search)
    fit_calls = MagicMock(return_value=_hand_posterior())
    monkeypatch.setattr("run_sizing.fit_posterior", fit_calls)
    monkeypatch.setattr("run_sizing.chosen_size", lambda *_a, **_k: 800)
    monkeypatch.setattr("run_sizing.should_stop", lambda **_k: should_stop)
    return fit_calls


def test_run_sizing_returns_typed_results(
    fake_photos: _FakePhotoSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_mocks(monkeypatch, should_stop=True)
    results = run_sizing(fake_photos.photos, seed_size=1)

    assert isinstance(results, SizingResults)
    assert isinstance(results.posterior_summary, PosteriorSummary)
    assert all(isinstance(r, PerPhotoResult) for r in results.per_photo)
    assert results.chosen_size_px == 800


def test_run_sizing_breaks_when_should_stop_fires_after_seed(
    fake_photos: _FakePhotoSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_mocks(monkeypatch, should_stop=True)
    results = run_sizing(fake_photos.photos, seed_size=1)
    # 3 photos available, but should_stop fires after the seed (n=1).
    assert len(results.per_photo) == 1


def test_run_sizing_skips_stop_check_under_seed_size(
    fake_photos: _FakePhotoSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """seed_size guard: fit_posterior shouldn't be called inside the loop
    until n_seen >= seed_size. It still gets one final call for the summary.
    """
    fit_calls = _wire_mocks(monkeypatch, should_stop=False)
    run_sizing(fake_photos.photos, seed_size=10)  # > photos available
    # Final summary fit only — none triggered inside the loop.
    assert fit_calls.call_count == 1


def test_run_sizing_processes_all_candidates_when_stop_never_fires(
    fake_photos: _FakePhotoSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a should_stop and with seed_size <= n_photos, every candidate
    is probed (no max-photos cap).
    """
    _wire_mocks(monkeypatch, should_stop=False)
    results = run_sizing(fake_photos.photos, seed_size=1)
    assert len(results.per_photo) == 3


def test_main_writes_results_json(
    fake_photos: _FakePhotoSet,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_path = tmp_path / "results.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_sizing.py",
            "--photo-dir",
            str(fake_photos.photo_dir),
            "--out",
            str(out_path),
            "--seed-size",
            "10",
        ],
    )
    _wire_mocks(monkeypatch, should_stop=False)

    rc = main()
    assert rc == 0
    payload = json.loads(out_path.read_text())
    # JSON-on-the-wire is dict-shaped, but the typed result was the source.
    assert payload["chosen_size_px"] == 800
    assert payload["n_candidate_photos"] == 3
    assert "DONE" in capsys.readouterr().err
