"""Tests for scripts/_pipeline.py — the pipeline-step decorator."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from _pipeline import (
    MissingInputError,
    StepDeclaration,
    clear_producers,
    declared_io,
    pipeline_step,
    register_producer,
)


@pytest.fixture(autouse=True)
def _isolate_producers() -> Iterator[None]:
    """Prevent test-to-test bleed-through of registered producers."""
    clear_producers()
    yield
    clear_producers()


# ---------------------------------------------------------------------------
# Happy path: declared inputs exist, function runs normally
# ---------------------------------------------------------------------------


def test_pipeline_step_runs_when_all_inputs_exist(tmp_path: Path) -> None:
    inp = tmp_path / "input.json"
    inp.write_text("{}")

    @pipeline_step(
        inputs=lambda x: [inp],
        outputs=lambda x: [],
    )
    def step(x: int) -> int:
        return x * 2

    assert step(21) == 42


def test_pipeline_step_resolves_inputs_per_call(tmp_path: Path) -> None:
    """Inputs callable receives the function's arguments — different
    calls can require different files.
    """
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("{}")
    b.write_text("{}")

    @pipeline_step(
        inputs=lambda which: [a if which == "a" else b],
        outputs=lambda which: [],
    )
    def step(which: str) -> str:
        return which

    assert step("a") == "a"
    assert step("b") == "b"


# ---------------------------------------------------------------------------
# Missing-input handling
# ---------------------------------------------------------------------------


def test_pipeline_step_raises_when_input_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-file.json"

    @pipeline_step(
        inputs=lambda: [missing],
        outputs=lambda: [],
    )
    def step() -> int:
        return 1  # pragma: no cover — wrapper raises before body runs

    with pytest.raises(MissingInputError) as exc:
        step()
    assert "missing input" in str(exc.value)
    assert str(missing) in str(exc.value)


def test_missing_input_error_includes_producer_hint(tmp_path: Path) -> None:
    """When the missing path matches a registered producer pattern,
    the error tells the user which command to run first.
    """
    register_producer("manifest.json", "scripts/group_pipeline.py <gid>")
    missing = tmp_path / "stitched_panels" / "g1" / "manifest.json"

    @pipeline_step(
        inputs=lambda: [missing],
        outputs=lambda: [],
    )
    def step() -> int:
        return 1  # pragma: no cover — wrapper raises before body runs

    with pytest.raises(MissingInputError) as exc:
        step()
    assert "produce it with: scripts/group_pipeline.py <gid>" in str(exc.value)


def test_missing_input_error_omits_hint_when_unknown(tmp_path: Path) -> None:
    missing = tmp_path / "no-rule-for-this.txt"

    @pipeline_step(
        inputs=lambda: [missing],
        outputs=lambda: [],
    )
    def step() -> int:
        return 1  # pragma: no cover — wrapper raises before body runs

    with pytest.raises(MissingInputError) as exc:
        step()
    assert "produce it with" not in str(exc.value)


def test_missing_input_short_circuits_function_call(tmp_path: Path) -> None:
    """If an input is missing, the wrapped function never runs.
    Important: pipeline steps may have side effects (writing files,
    making LLM calls) and we don't want a partial run on missing inputs.
    """
    missing = tmp_path / "absent.json"
    called = {"count": 0}

    @pipeline_step(
        inputs=lambda: [missing],
        outputs=lambda: [],
    )
    def step() -> int:
        called["count"] += 1  # pragma: no cover
        return 1  # pragma: no cover

    with pytest.raises(MissingInputError):
        step()
    assert called["count"] == 0


def test_pipeline_step_message_uses_explicit_name(tmp_path: Path) -> None:
    """The decorator's ``name=`` kwarg overrides the qualname for
    cleaner error messages on long-named functions.
    """
    missing = tmp_path / "absent.json"

    @pipeline_step(
        inputs=lambda: [missing],
        outputs=lambda: [],
        name="extract_yaml",
    )
    def some_long_qualified_name() -> int:
        return 1  # pragma: no cover — wrapper raises before body runs

    with pytest.raises(MissingInputError) as exc:
        some_long_qualified_name()
    assert "[extract_yaml]" in str(exc.value)


def test_eager_input_check_off_lets_function_handle_missing(
    tmp_path: Path,
) -> None:
    """With ``eager_input_check=False`` the wrapper does NOT check
    inputs; the function decides for itself. Useful for steps with a
    skip-when-output-exists fast path that shouldn't block on stale
    inputs that aren't actually consumed.
    """
    missing = tmp_path / "absent.json"

    @pipeline_step(
        inputs=lambda: [missing],
        outputs=lambda: [],
        eager_input_check=False,
    )
    def step() -> int:
        return 42

    assert step() == 42


def test_missing_input_error_is_a_filenotfounderror(tmp_path: Path) -> None:
    """Catchable as the standard exception so pipelines using
    try/except FileNotFoundError still work, but the str() form
    keeps the producer hint.
    """
    missing = tmp_path / "absent.json"

    @pipeline_step(
        inputs=lambda: [missing],
        outputs=lambda: [],
    )
    def step() -> int:
        return 1  # pragma: no cover — wrapper raises before body runs

    with pytest.raises(FileNotFoundError):
        step()


# ---------------------------------------------------------------------------
# Producer registry
# ---------------------------------------------------------------------------


def test_register_producer_first_match_wins(tmp_path: Path) -> None:
    """Earlier-registered patterns win on conflict — register more
    specific patterns first.
    """
    register_producer("stitched_panels/foo/manifest.json", "scripts/special.py")
    register_producer("manifest.json", "scripts/general.py")
    missing = tmp_path / "stitched_panels" / "foo" / "manifest.json"

    @pipeline_step(
        inputs=lambda: [missing],
        outputs=lambda: [],
    )
    def step() -> int:
        return 1  # pragma: no cover — wrapper raises before body runs

    with pytest.raises(MissingInputError) as exc:
        step()
    assert "scripts/special.py" in str(exc.value)
    assert "scripts/general.py" not in str(exc.value)


def test_clear_producers_drops_all(tmp_path: Path) -> None:
    register_producer("anything", "scripts/foo.py")
    clear_producers()
    missing = tmp_path / "anything-here.json"

    @pipeline_step(
        inputs=lambda: [missing],
        outputs=lambda: [],
    )
    def step() -> int:
        return 1  # pragma: no cover — wrapper raises before body runs

    with pytest.raises(MissingInputError) as exc:
        step()
    assert "produce it with" not in str(exc.value)


# ---------------------------------------------------------------------------
# declared_io introspection
# ---------------------------------------------------------------------------


def test_declared_io_returns_inputs_and_outputs(tmp_path: Path) -> None:
    """The runs/ recorder (next commit) calls declared_io to capture a
    step's I/O without invoking it.
    """
    inp = tmp_path / "i.json"
    out = tmp_path / "o.json"

    @pipeline_step(
        inputs=lambda x: [inp],
        outputs=lambda x: [out],
    )
    def step(x: int) -> int:
        return x  # pragma: no cover — declared_io doesn't invoke

    decl = declared_io(step, 7)
    assert decl == StepDeclaration(inputs=(inp,), outputs=(out,))


def test_declared_io_rejects_undecorated_function() -> None:
    def plain(x: int) -> int:
        return x  # pragma: no cover — declared_io rejects before invoking

    with pytest.raises(ValueError, match="not a @pipeline_step"):
        declared_io(plain, 1)
