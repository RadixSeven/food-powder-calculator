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
from run_sizing import list_nutrition_photos, main, run_sizing
from sizing_model import Posterior, Probe


@dataclass
class _FakePhotoSet:
    """Holds a fake gold_groups.json on disk plus the resolved photo paths."""

    gold_groups_json: Path
    nutrition_paths: list[Path]


@pytest.fixture
def fake_gold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakePhotoSet:
    """Build a minimal gold_groups.json with two nutrition photos and one
    non-nutrition photo (which the runner should skip)."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    nutrition_paths = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        p = photos_dir / name
        p.write_bytes(b"x")
        nutrition_paths.append(p)
    gold_groups_json = tmp_path / "gold.json"
    gold_groups_json.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "g1",
                        "store": "MOM",
                        "photos": [
                            {
                                "path": str(nutrition_paths[0]),
                                "roles": ["nutrition"],
                            },
                            {
                                "path": str(nutrition_paths[1]),
                                "roles": ["front"],
                            },
                            {
                                "path": str(nutrition_paths[2]),
                                "roles": ["nutrition", "ingredients"],
                            },
                        ],
                        "warnings": [],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr("run_sizing.REPO_ROOT", tmp_path)
    return _FakePhotoSet(
        gold_groups_json=gold_groups_json,
        nutrition_paths=[nutrition_paths[0], nutrition_paths[2]],
    )


def test_list_nutrition_photos_filters_by_role(
    fake_gold: _FakePhotoSet,
) -> None:
    found = list_nutrition_photos(fake_gold.gold_groups_json)
    assert len(found) == 2
    assert all(p.exists() for p in found)


def _hand_posterior(
    mu_mean: float = 6.5, sigma_mean: float = 0.3, n_photos: int = 5
) -> Posterior:
    return Posterior(
        mu_samples=np.full(200, mu_mean),
        sigma_samples=np.full(200, sigma_mean),
        n_photos_observed=n_photos,
    )


def test_run_sizing_stops_at_max_photos_cap(
    fake_gold: _FakePhotoSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With max_photos lower than the candidate count, the loop stops at the
    cap regardless of what should_stop says."""

    def fake_extract(*_a: object, **_k: object) -> ExtractedPayload:
        return ExtractedPayload(text="hello", barcodes=())

    def fake_select(*_a: object, **_k: object) -> tuple[str, ExtractedPayload]:
        return "haiku", ExtractedPayload(text="hello", barcodes=())

    def fake_search(*_a: object, **k: object) -> SearchResult:
        pid = cast(str, k["photo_id"])
        return SearchResult(
            photo_id=pid,
            reference_model="opus",
            search_model="haiku",
            min_legible_size=512,
            probes=[Probe(photo_id=pid, size=512, matched=True)],
        )

    monkeypatch.setattr("run_sizing.extract_payload", fake_extract)
    monkeypatch.setattr("run_sizing.select_search_model", fake_select)
    monkeypatch.setattr("run_sizing.binary_search_min_size", fake_search)
    monkeypatch.setattr(
        "run_sizing.fit_posterior", lambda probes, **_k: _hand_posterior()
    )
    monkeypatch.setattr("run_sizing.should_stop", lambda **_k: False)

    candidates = fake_gold.nutrition_paths
    results = run_sizing(candidates, seed_size=10, max_photos=1)

    assert len(cast(list[object], results["per_photo"])) == 1
    assert "chosen_size_px" in results
    assert "posterior_summary" in results


def test_run_sizing_breaks_when_should_stop_fires(
    fake_gold: _FakePhotoSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """should_stop returning True after the seed exits the loop early."""

    monkeypatch.setattr(
        "run_sizing.extract_payload",
        lambda *_a, **_k: ExtractedPayload(text="hi", barcodes=()),
    )
    monkeypatch.setattr(
        "run_sizing.select_search_model",
        lambda *_a, **_k: ("haiku", ExtractedPayload(text="hi", barcodes=())),
    )

    def fake_search(*_a: object, **k: object) -> SearchResult:
        pid = cast(str, k["photo_id"])
        return SearchResult(
            photo_id=pid,
            reference_model="opus",
            search_model="haiku",
            min_legible_size=512,
            probes=[Probe(photo_id=pid, size=512, matched=True)],
        )

    monkeypatch.setattr("run_sizing.binary_search_min_size", fake_search)
    monkeypatch.setattr(
        "run_sizing.fit_posterior", lambda probes, **_k: _hand_posterior()
    )
    monkeypatch.setattr("run_sizing.chosen_size", lambda *_a, **_k: 800)
    monkeypatch.setattr("run_sizing.should_stop", lambda **_k: True)

    candidates = fake_gold.nutrition_paths
    results = run_sizing(candidates, seed_size=1, max_photos=10)

    assert len(cast(list[object], results["per_photo"])) == 1


def test_run_sizing_skips_stop_check_under_seed_size(
    fake_gold: _FakePhotoSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """seed_size guard: fit_posterior shouldn't be called until n_seen >=
    seed_size."""

    monkeypatch.setattr(
        "run_sizing.extract_payload",
        lambda *_a, **_k: ExtractedPayload(text="hi", barcodes=()),
    )
    monkeypatch.setattr(
        "run_sizing.select_search_model",
        lambda *_a, **_k: ("haiku", ExtractedPayload(text="hi", barcodes=())),
    )

    def fake_search(*_a: object, **k: object) -> SearchResult:
        pid = cast(str, k["photo_id"])
        return SearchResult(
            photo_id=pid,
            reference_model="opus",
            search_model="haiku",
            min_legible_size=512,
            probes=[Probe(photo_id=pid, size=512, matched=True)],
        )

    monkeypatch.setattr("run_sizing.binary_search_min_size", fake_search)

    fit_calls = MagicMock(return_value=_hand_posterior())
    monkeypatch.setattr("run_sizing.fit_posterior", fit_calls)
    monkeypatch.setattr("run_sizing.should_stop", lambda **_k: False)

    candidates = fake_gold.nutrition_paths
    run_sizing(candidates, seed_size=10, max_photos=2)
    # fit_posterior should be called once at the very end (final summary),
    # not during the per-photo loop because n_seen never reached seed_size.
    assert fit_calls.call_count == 1


def test_main_writes_results_json(
    fake_gold: _FakePhotoSet,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_path = tmp_path / "results.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_sizing.py",
            "--gold-groups",
            str(fake_gold.gold_groups_json),
            "--out",
            str(out_path),
            "--max-photos",
            "1",
            "--seed-size",
            "10",
        ],
    )
    monkeypatch.setattr(
        "run_sizing.extract_payload",
        lambda *_a, **_k: ExtractedPayload(text="hi", barcodes=()),
    )
    monkeypatch.setattr(
        "run_sizing.select_search_model",
        lambda *_a, **_k: ("haiku", ExtractedPayload(text="hi", barcodes=())),
    )

    def fake_search(*_a: object, **k: object) -> SearchResult:
        pid = cast(str, k["photo_id"])
        return SearchResult(
            photo_id=pid,
            reference_model="opus",
            search_model="haiku",
            min_legible_size=512,
            probes=[Probe(photo_id=pid, size=512, matched=True)],
        )

    monkeypatch.setattr("run_sizing.binary_search_min_size", fake_search)
    monkeypatch.setattr(
        "run_sizing.fit_posterior", lambda probes, **_k: _hand_posterior()
    )
    monkeypatch.setattr("run_sizing.chosen_size", lambda *_a, **_k: 800)
    monkeypatch.setattr("run_sizing.should_stop", lambda **_k: False)

    rc = main()
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert payload["chosen_size_px"] == 800
    assert payload["n_candidate_photos"] == 2
    out = capsys.readouterr().err
    assert "DONE" in out
