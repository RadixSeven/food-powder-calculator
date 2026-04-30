"""Tests for scripts/provenance.py — the run-lookup CLI."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from provenance import find_runs_producing, format_matches, main


def _write_manifest(
    runs_dir: Path,
    *,
    run_id: str,
    started_at: str,
    output_path: str,
    output_sha: str = "deadbeef",
    n_inputs: int = 0,
    step_name: str = "test-step",
    git_sha: str = "abc1234567890",
) -> Path:
    """Drop a synthetic manifest under runs_dir/run_id/manifest.yaml."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "run_id": run_id,
        "git_sha": git_sha,
        "git_dirty": False,
        "started_at": started_at,
        "ended_at": started_at,
        "argv": ["scripts/group_pipeline.py", "x"],
        "env": cast("dict[str, object]", {}),
        "steps": [
            {
                "name": step_name,
                "started_at": started_at,
                "ended_at": started_at,
                "inputs": [
                    {"path": f"in{i}", "sha256": ""} for i in range(n_inputs)
                ],
                "outputs": [{"path": output_path, "sha256": output_sha}],
            }
        ],
        "calls": cast("list[object]", []),
    }
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump(payload))
    return run_dir / "manifest.yaml"


def test_find_runs_producing_returns_match(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    target = tmp_path / "out.json"
    _write_manifest(
        runs_dir,
        run_id="2026-04-30T10-00-00-aaa",
        started_at="2026-04-30T10:00:00-04:00",
        output_path=str(target),
        n_inputs=4,
        step_name="group_pipeline.process_group",
    )
    matches = find_runs_producing(target, runs_dir=runs_dir)
    assert len(matches) == 1
    assert matches[0].run_id == "2026-04-30T10-00-00-aaa"
    assert matches[0].step_name == "group_pipeline.process_group"
    assert matches[0].n_inputs == 4
    assert matches[0].matched_path == str(target)


def test_find_runs_producing_returns_empty_for_unknown_path(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    matches = find_runs_producing(tmp_path / "no-such.json", runs_dir=runs_dir)
    assert matches == []


def test_find_runs_producing_returns_empty_for_missing_runs_dir(
    tmp_path: Path,
) -> None:
    matches = find_runs_producing(
        tmp_path / "target.json", runs_dir=tmp_path / "absent"
    )
    assert matches == []


def test_find_runs_producing_sorts_most_recent_first(tmp_path: Path) -> None:
    """Multiple runs produced the same path (e.g. a file rebuilt across
    runs); the most recent wins so callers can take the head of the
    list as the answer.
    """
    runs_dir = tmp_path / "runs"
    target = tmp_path / "out.json"
    _write_manifest(
        runs_dir,
        run_id="2026-04-29T10-00-00-aaa",
        started_at="2026-04-29T10:00:00-04:00",
        output_path=str(target),
    )
    _write_manifest(
        runs_dir,
        run_id="2026-04-30T10-00-00-bbb",
        started_at="2026-04-30T10:00:00-04:00",
        output_path=str(target),
    )
    matches = find_runs_producing(target, runs_dir=runs_dir)
    assert [m.run_id for m in matches] == [
        "2026-04-30T10-00-00-bbb",
        "2026-04-29T10-00-00-aaa",
    ]


def test_find_runs_producing_skips_manifests_with_wrong_shape(
    tmp_path: Path,
) -> None:
    """A manifest at the right path but with an unexpected JSON shape
    (top-level array, missing steps, garbled step entries) is skipped
    rather than crashing the scan — defensive against partial / hand-
    edited files.
    """
    runs_dir = tmp_path / "runs"
    target = tmp_path / "out.json"
    bad_dir = runs_dir / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "manifest.yaml").write_text("- just a list\n")

    no_steps = runs_dir / "no-steps"
    no_steps.mkdir(parents=True)
    (no_steps / "manifest.yaml").write_text(yaml.safe_dump({"run_id": "x"}))

    garbled_step = runs_dir / "garbled-step"
    garbled_step.mkdir(parents=True)
    (garbled_step / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "x",
                "started_at": "t",
                "steps": ["not a dict", {"outputs": "not a list"}],
            }
        )
    )

    # And one valid manifest that should match.
    _write_manifest(
        runs_dir,
        run_id="good",
        started_at="2026-04-30T10:00:00-04:00",
        output_path=str(target),
    )
    matches = find_runs_producing(target, runs_dir=runs_dir)
    assert [m.run_id for m in matches] == ["good"]


def test_find_runs_producing_handles_garbled_output_entries(
    tmp_path: Path,
) -> None:
    """Output rows with unexpected shapes are skipped per-row so a
    single bad output doesn't block other matches in the same step.
    """
    runs_dir = tmp_path / "runs"
    target = tmp_path / "out.json"
    run_dir = runs_dir / "r"
    run_dir.mkdir(parents=True)
    payload: dict[str, object] = {
        "run_id": "r",
        "started_at": "2026-04-30T10:00:00-04:00",
        "steps": [
            {
                "name": "s",
                "started_at": "t",
                "ended_at": "t",
                "inputs": [],
                "outputs": [
                    "not a dict",
                    {"path": str(target), "sha256": "abcd"},
                ],
            }
        ],
    }
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump(payload))
    matches = find_runs_producing(target, runs_dir=runs_dir)
    assert len(matches) == 1
    assert matches[0].matched_sha256 == "abcd"


def test_find_runs_skips_non_matching_outputs_within_step(
    tmp_path: Path,
) -> None:
    """A step's outputs list often contains several files; only the
    matching one returns. Ensures the per-output `continue` path
    runs when the path doesn't match.
    """
    runs_dir = tmp_path / "runs"
    target = tmp_path / "wanted.json"
    other = tmp_path / "other.json"
    run_dir = runs_dir / "r"
    run_dir.mkdir(parents=True)
    payload: dict[str, object] = {
        "run_id": "r",
        "started_at": "2026-04-30T10:00:00-04:00",
        "steps": [
            {
                "name": "s",
                "started_at": "t",
                "ended_at": "t",
                "inputs": [],
                "outputs": [
                    {"path": str(other), "sha256": "1111"},
                    {"path": str(target), "sha256": "2222"},
                ],
            }
        ],
    }
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump(payload))
    matches = find_runs_producing(target, runs_dir=runs_dir)
    assert len(matches) == 1
    assert matches[0].matched_sha256 == "2222"


def test_format_matches_no_matches_message(tmp_path: Path) -> None:
    text = format_matches(tmp_path / "x", [])
    assert "No runs" in text
    assert "data/runs/" in text


def test_format_matches_renders_each_match(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    target = tmp_path / "out.json"
    _write_manifest(
        runs_dir,
        run_id="run-1",
        started_at="2026-04-30T10:00:00-04:00",
        output_path=str(target),
        n_inputs=3,
        step_name="extract_yaml.process_group_to_yaml",
    )
    matches = find_runs_producing(target, runs_dir=runs_dir)
    text = format_matches(target, matches)
    assert "run-1" in text
    assert "extract_yaml.process_group_to_yaml" in text
    assert "inputs:     3" in text


def test_main_returns_zero_on_match(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    runs_dir = tmp_path / "runs"
    target = tmp_path / "out.json"
    _write_manifest(
        runs_dir,
        run_id="r",
        started_at="2026-04-30T10:00:00-04:00",
        output_path=str(target),
    )

    import sys

    monkeypatch.setattr(  # type: ignore[attr-defined]
        sys,
        "argv",
        [
            "provenance.py",
            str(target),
            "--runs-dir",
            str(runs_dir),
        ],
    )
    rc = main()
    assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "run-id" in captured.out.lower() or "run_id" in captured.out


def test_main_returns_one_on_no_match(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    """Exit code 1 lets shell users distinguish "no provenance" from
    "found provenance" without parsing stdout.
    """
    import sys

    monkeypatch.setattr(  # type: ignore[attr-defined]
        sys,
        "argv",
        [
            "provenance.py",
            str(tmp_path / "absent.json"),
            "--runs-dir",
            str(tmp_path / "no-runs"),
        ],
    )
    rc = main()
    assert rc == 1
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "No runs" in captured.out
