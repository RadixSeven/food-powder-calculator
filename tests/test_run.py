"""Tests for scripts/_run.py — per-invocation run records."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from _run import (
    CallRecord,
    Run,
    StepRecord,
    current_run,
    open_run,
    sha256_of_file,
)


def test_open_run_writes_manifest_with_minimum_fields(
    tmp_path: Path,
) -> None:
    """A run with no steps and no calls still produces a manifest
    capturing identity + timing + env. The provenance lookup keys off
    the manifest's mere existence, so the empty case has to work.
    """
    runs_dir = tmp_path / "runs"
    fixed_now = datetime.fromisoformat("2026-04-30T14:23:01+00:00")
    with patch("_run._git_sha", return_value="abc1234567890"):
        with patch("_run._git_is_dirty", return_value=False):
            with open_run(
                argv=["scripts/foo.py", "--bar"],
                runs_dir=runs_dir,
                now=fixed_now,
            ) as run:
                pass
    manifest = run.directory / "manifest.yaml"
    assert manifest.exists()
    payload = yaml.safe_load(manifest.read_text())
    assert payload["run_id"].startswith("2026-04-30T14-23-01-abc1234")
    assert payload["git_sha"] == "abc1234567890"
    assert payload["git_dirty"] is False
    assert payload["argv"] == ["scripts/foo.py", "--bar"]
    assert payload["steps"] == []
    assert payload["calls"] == []
    assert "python_version" in payload["env"]


def test_open_run_appends_counter_on_collision(tmp_path: Path) -> None:
    """Two runs at the same second on the same git SHA must end up
    with distinct directories — append ``-1``, ``-2`` to the run_id.
    """
    runs_dir = tmp_path / "runs"
    fixed_now = datetime.fromisoformat("2026-04-30T14:23:01+00:00")
    with patch("_run._git_sha", return_value="abc1234567890"):
        with patch("_run._git_is_dirty", return_value=False):
            with open_run(argv=["x"], runs_dir=runs_dir, now=fixed_now) as run1:
                pass
            with open_run(argv=["x"], runs_dir=runs_dir, now=fixed_now) as run2:
                pass
            with open_run(argv=["x"], runs_dir=runs_dir, now=fixed_now) as run3:
                pass
    assert run1.run_id != run2.run_id != run3.run_id
    assert run2.run_id.endswith("-1")
    assert run3.run_id.endswith("-2")


def test_open_run_handles_missing_git(tmp_path: Path) -> None:
    """A run outside a git repo (or with git missing) still works;
    git_sha becomes empty and the run_id falls back to a sortable
    timestamp + ``no-git``.
    """
    runs_dir = tmp_path / "runs"
    fixed_now = datetime.fromisoformat("2026-04-30T14:23:01+00:00")
    with patch("_run._git_sha", return_value=None):
        with patch("_run._git_is_dirty", return_value=False):
            with open_run(argv=["x"], runs_dir=runs_dir, now=fixed_now) as run:
                pass
    payload = yaml.safe_load((run.directory / "manifest.yaml").read_text())
    assert payload["git_sha"] == ""
    assert "no-git" in run.run_id


def test_open_run_writes_manifest_even_on_exception(
    tmp_path: Path,
) -> None:
    """If the contained code raises, the manifest is still written so
    the partial run is auditable. The exception then propagates.
    """
    runs_dir = tmp_path / "runs"
    with patch("_run._git_sha", return_value="abc1234567890"):
        with patch("_run._git_is_dirty", return_value=False):
            with pytest.raises(RuntimeError, match="boom"):
                with open_run(argv=["x"], runs_dir=runs_dir) as run:
                    run.record_step(
                        StepRecord(
                            name="thing",
                            started_at="t1",
                            ended_at="t2",
                            inputs=(("a", "sha-a"),),
                            outputs=(("b", "sha-b"),),
                        )
                    )
                    raise RuntimeError("boom")
    manifest = run.directory / "manifest.yaml"
    assert manifest.exists()
    payload = yaml.safe_load(manifest.read_text())
    assert payload["steps"][0]["name"] == "thing"


def test_record_step_appends_to_run() -> None:
    run = Run(
        run_id="r",
        git_sha="",
        git_dirty=False,
        argv=(),
        started_at="t0",
        runs_dir=Path("."),
    )
    run.record_step(
        StepRecord(
            name="s",
            started_at="t1",
            ended_at="t2",
            inputs=(),
            outputs=(),
        )
    )
    assert len(run.steps) == 1


def test_record_call_appends_and_tracks_models() -> None:
    """Each call records the model; the manifest's env.models_used
    summarizes the set so the user can see at a glance that this run
    used only haiku, etc.
    """
    run = Run(
        run_id="r",
        git_sha="",
        git_dirty=False,
        argv=(),
        started_at="t0",
        runs_dir=Path("."),
    )
    run.record_call(sha="abc", cached=True, elapsed_seconds=0.0, model="haiku")
    run.record_call(sha="def", cached=False, elapsed_seconds=4.2, model="opus")
    assert run.calls == [
        CallRecord(sha="abc", cached=True, elapsed_seconds=0.0),
        CallRecord(sha="def", cached=False, elapsed_seconds=4.2),
    ]
    assert run.models_used == {"haiku", "opus"}


def test_manifest_serializes_steps_and_calls(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    with patch("_run._git_sha", return_value="abc1234567890"):
        with patch("_run._git_is_dirty", return_value=False):
            with open_run(argv=["x"], runs_dir=runs_dir) as run:
                run.record_step(
                    StepRecord(
                        name="s1",
                        started_at="t1",
                        ended_at="t2",
                        inputs=(("/a", "sha-a"),),
                        outputs=(("/b", "sha-b"),),
                    )
                )
                run.record_call(
                    sha="aaa",
                    cached=True,
                    elapsed_seconds=0.0,
                    model="haiku",
                )
                run.record_call(
                    sha="bbb",
                    cached=False,
                    elapsed_seconds=4.213,
                    model="opus",
                )
    payload = yaml.safe_load((run.directory / "manifest.yaml").read_text())
    assert payload["steps"] == [
        {
            "name": "s1",
            "started_at": "t1",
            "ended_at": "t2",
            "inputs": [{"path": "/a", "sha256": "sha-a"}],
            "outputs": [{"path": "/b", "sha256": "sha-b"}],
        }
    ]
    assert payload["calls"][0] == {
        "sha": "aaa",
        "cached": True,
        "elapsed_seconds": 0.0,
    }
    assert payload["calls"][1]["elapsed_seconds"] == 4.213
    assert payload["env"]["models_used"] == ["haiku", "opus"]


def test_current_run_is_set_inside_context(tmp_path: Path) -> None:
    """Inside ``open_run`` the contextvar points at the active run;
    outside it's None. @pipeline_step and _claude.call rely on this.
    """
    runs_dir = tmp_path / "runs"
    assert current_run.get() is None
    with patch("_run._git_sha", return_value="abc1234567890"):
        with patch("_run._git_is_dirty", return_value=False):
            with open_run(argv=["x"], runs_dir=runs_dir) as run:
                assert current_run.get() is run
    assert current_run.get() is None


def test_sha256_of_file_returns_hex(tmp_path: Path) -> None:
    p = tmp_path / "file.txt"
    p.write_bytes(b"hello world")
    # Known SHA-256 of "hello world".
    assert (
        sha256_of_file(p)
        == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_sha256_of_file_returns_empty_for_missing(tmp_path: Path) -> None:
    """Used by step recording when an output was meant to be written
    but isn't there for some reason — record the path with empty SHA
    instead of raising.
    """
    assert sha256_of_file(tmp_path / "no-such-file") == ""


def test_pipeline_step_records_into_open_run(tmp_path: Path) -> None:
    """End-to-end: a @pipeline_step invocation inside open_run shows up
    in the manifest with its declared inputs/outputs and their SHAs.
    """
    from _pipeline import pipeline_step

    inp = tmp_path / "input.json"
    out = tmp_path / "output.json"
    inp.write_text("{}")

    @pipeline_step(
        inputs=lambda: [inp],
        outputs=lambda: [out],
        name="fake-step",
    )
    def step() -> None:
        out.write_text("done")

    runs_dir = tmp_path / "runs"
    with patch("_run._git_sha", return_value="abc1234567890"):
        with patch("_run._git_is_dirty", return_value=False):
            with open_run(argv=["x"], runs_dir=runs_dir) as run:
                step()
    payload = yaml.safe_load((run.directory / "manifest.yaml").read_text())
    assert len(payload["steps"]) == 1
    s = payload["steps"][0]
    assert s["name"] == "fake-step"
    assert len(s["inputs"]) == 1
    assert s["inputs"][0]["path"] == str(inp)
    assert s["inputs"][0]["sha256"]
    assert s["outputs"][0]["path"] == str(out)
    assert s["outputs"][0]["sha256"]


def test_pipeline_step_records_step_when_function_raises(
    tmp_path: Path,
) -> None:
    """Even if the step raises, the manifest captures what was
    attempted (with SHAs of inputs as they were at run time, and
    outputs whose SHAs may be empty if the step never wrote them).
    """
    from _pipeline import pipeline_step

    inp = tmp_path / "input.json"
    out = tmp_path / "output.json"
    inp.write_text("{}")

    @pipeline_step(
        inputs=lambda: [inp],
        outputs=lambda: [out],
        name="failing-step",
    )
    def step() -> None:
        raise RuntimeError("boom")

    runs_dir = tmp_path / "runs"
    with patch("_run._git_sha", return_value="abc1234567890"):
        with patch("_run._git_is_dirty", return_value=False):
            with pytest.raises(RuntimeError, match="boom"):
                with open_run(argv=["x"], runs_dir=runs_dir) as run:
                    step()
    payload = yaml.safe_load((run.directory / "manifest.yaml").read_text())
    s = payload["steps"][0]
    assert s["name"] == "failing-step"
    # Output never got written; sha is empty but path is recorded.
    assert s["outputs"][0]["sha256"] == ""


def test_pipeline_step_outside_run_does_not_crash(tmp_path: Path) -> None:
    """Running a @pipeline_step outside any open_run is fine — the
    hooks no-op when current_run is None. Important for tests and
    ad-hoc invocations.
    """
    from _pipeline import pipeline_step

    inp = tmp_path / "input.json"
    inp.write_text("{}")
    called = {"count": 0}

    @pipeline_step(inputs=lambda: [inp], outputs=lambda: [])
    def step() -> int:
        called["count"] += 1
        return 7

    assert step() == 7
    assert called["count"] == 1


def test_git_sha_returns_stdout_on_success() -> None:
    """The git subprocess wrapper returns trimmed stdout when git is
    happy. Patches subprocess.run so the test doesn't depend on the
    repo we're actually running in.
    """
    from _run import _git_sha

    class FakeResult:
        returncode = 0
        stdout = "abc1234567890\n"
        stderr = ""

    with patch("_run.subprocess.run", return_value=FakeResult()):
        assert _git_sha() == "abc1234567890"


def test_git_sha_returns_none_on_failure() -> None:
    from _run import _git_sha

    class FakeResult:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository\n"

    with patch("_run.subprocess.run", return_value=FakeResult()):
        assert _git_sha() is None


def test_git_is_dirty_true_when_porcelain_nonempty() -> None:
    from _run import _git_is_dirty

    class FakeResult:
        returncode = 0
        stdout = " M scripts/foo.py\n"
        stderr = ""

    with patch("_run.subprocess.run", return_value=FakeResult()):
        assert _git_is_dirty() is True


def test_git_is_dirty_false_when_clean() -> None:
    from _run import _git_is_dirty

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    with patch("_run.subprocess.run", return_value=FakeResult()):
        assert _git_is_dirty() is False


def test_git_is_dirty_false_on_subprocess_failure() -> None:
    from _run import _git_is_dirty

    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "error"

    with patch("_run.subprocess.run", return_value=FakeResult()):
        assert _git_is_dirty() is False


def test_claude_call_records_into_open_run(tmp_path: Path) -> None:
    """End-to-end: an LLM call inside open_run shows up in the
    manifest's calls list. Mock _run_claude_subprocess to avoid
    network access.
    """
    from _claude import ClaudeRequest, call

    runs_dir = tmp_path / "runs"
    cache_dir = tmp_path / "cache"
    with patch("_run._git_sha", return_value="abc1234567890"):
        with patch("_run._git_is_dirty", return_value=False):
            with patch("_claude.CACHE_DIR", cache_dir):
                with patch(
                    "_claude._run_claude_subprocess", return_value="api"
                ):
                    with open_run(argv=["x"], runs_dir=runs_dir) as run:
                        call(ClaudeRequest(prompt="hi", model="haiku"))
    payload = yaml.safe_load((run.directory / "manifest.yaml").read_text())
    assert len(payload["calls"]) == 1
    assert payload["calls"][0]["cached"] is False
    assert payload["env"]["models_used"] == ["haiku"]
