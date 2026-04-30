"""Pipeline-step infrastructure: declared inputs/outputs + missing-input errors.

Each pipeline command (group_pipeline.process_group, extract_yaml.
process_group_to_yaml, etc.) declares the durable artifacts it
consumes and produces. The :func:`pipeline_step` decorator:

1. Verifies every declared input exists on disk before running. The
   error message points at the command registered as the producer of
   that path, so a missing input becomes a "run X first" instruction
   rather than a deep stack trace.
2. (Future, in the runs/ schema commit) Captures the (inputs,
   outputs) tuple so a separate run-recorder can write
   ``data/runs/<id>/manifest.yaml``.

The producer registry is intentionally simple: a pattern → command
mapping, where the pattern is a substring match against the path's
string form. A path matches the first registered pattern that's a
substring. That's enough for our paths (e.g.
``data/stitched_panels/<gid>/manifest.json`` matches the substring
``stitched_panels``); proper glob support can come later if we need
it.

Cache paths must NEVER appear in declared inputs/outputs — see the
``data/cache/`` discussion in scripts/_claude.py. Inputs are durable
by construction; the wrapper enforces that the file was on disk
before the command ran, regardless of whether it was just produced
by a previous step or has been there for years.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# Registry of "this path pattern is produced by this command". The
# pattern is matched against ``str(path)`` via substring. First match
# wins, in registration order — register more-specific patterns first.
_PRODUCER_REGISTRY: list[tuple[str, str]] = []


@dataclass(frozen=True)
class StepDeclaration:
    """The declared inputs/outputs for one pipeline-step invocation.

    Resolved at call time from the user-supplied callables, so a
    step's I/O is always a function of its arguments. Returned by
    the wrapper's metadata-capture hooks so a future runs/ recorder
    can include it without re-running the I/O resolvers.
    """

    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]


class MissingInputError(FileNotFoundError):
    """Raised when a declared input doesn't exist on disk.

    Subclasses ``FileNotFoundError`` so callers can use the standard
    exception type to handle "missing file" failures while still
    preserving the producer-hint message in ``str(exc)``.
    """


def register_producer(pattern: str, command: str) -> None:
    """Register that paths containing ``pattern`` are produced by ``command``.

    The next time :func:`pipeline_step` finds a missing input whose
    path string contains ``pattern``, the raised
    :class:`MissingInputError` will tell the user to run ``command``
    to produce it. Multiple patterns may match a given path; the
    first registered wins, so register more-specific patterns first.
    """
    _PRODUCER_REGISTRY.append((pattern, command))


def clear_producers() -> None:
    """Drop all registered producers — for tests only."""
    _PRODUCER_REGISTRY.clear()


def pipeline_step(
    *,
    inputs: Callable[P, Sequence[Path]],
    outputs: Callable[P, Sequence[Path]],
    name: str | None = None,
    eager_input_check: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Build a decorator that verifies declared inputs exist before the call.

    Both ``inputs`` and ``outputs`` are callables that take the
    decorated function's arguments and return the relevant durable
    paths. They're called once per invocation. Outputs aren't checked
    here (they're what the function produces) — they're declared so a
    future runs/ recorder can capture them.

    ``name`` defaults to the wrapped function's qualified name; pass
    a shorter explicit name when the qualname is awkward in error
    messages.

    ``eager_input_check`` (default ``True``): verify every input
    exists before calling the function. Set ``False`` for functions
    with a skip-when-output-exists fast path — the wrapped function
    can decide for itself when inputs matter.

    On a missing input (when checking is enabled), raises
    :class:`MissingInputError` with a message that includes the
    missing path and (if registered) the command that produces it.
    The function never runs in that case.
    """

    def wrap(fn: Callable[P, R]) -> Callable[P, R]:
        step_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if eager_input_check:
                declared_inputs = tuple(inputs(*args, **kwargs))
                for inp in declared_inputs:
                    if not inp.exists():
                        raise MissingInputError(
                            _missing_input_message(step_name, inp)
                        )
            # Capture started/ended timestamps and (path, sha256) pairs
            # for inputs and outputs so an open run (if any) can record
            # the step. Imported lazily to avoid a circular import:
            # _run imports nothing from _pipeline today, but a future
            # tightening could.
            from _run import current_run, sha256_of_file  # noqa: PLC0415

            run = current_run.get()
            started = datetime.now().astimezone().isoformat(timespec="seconds")
            try:
                result = fn(*args, **kwargs)
            finally:
                ended = (
                    datetime.now().astimezone().isoformat(timespec="seconds")
                )
                if run is not None:
                    record_inputs = tuple(
                        (str(p), sha256_of_file(p))
                        for p in inputs(*args, **kwargs)
                    )
                    record_outputs = tuple(
                        (str(p), sha256_of_file(p))
                        for p in outputs(*args, **kwargs)
                    )
                    from _run import StepRecord  # noqa: PLC0415

                    run.record_step(
                        StepRecord(
                            name=step_name,
                            started_at=started,
                            ended_at=ended,
                            inputs=record_inputs,
                            outputs=record_outputs,
                        )
                    )
            return result

        # Expose the resolvers for callers that want to introspect
        # what a step would consume/produce without invoking it
        # (e.g. the runs/ recorder, or a "what's missing?" check).
        wrapper.__pipeline_step_name__ = step_name  # type: ignore[attr-defined]
        wrapper.__pipeline_inputs__ = inputs  # type: ignore[attr-defined]
        wrapper.__pipeline_outputs__ = outputs  # type: ignore[attr-defined]
        return wrapper

    return wrap


def declared_io(
    fn: Callable[..., object], *args: object, **kwargs: object
) -> StepDeclaration:
    """Resolve a step's declared inputs/outputs for a given call.

    Used by the runs/ recorder (and by tests / introspection). Raises
    ``ValueError`` if ``fn`` isn't a ``@pipeline_step``-decorated
    function — there's no good fallback for an undecorated callable.
    """
    inputs_resolver = getattr(fn, "__pipeline_inputs__", None)
    outputs_resolver = getattr(fn, "__pipeline_outputs__", None)
    if inputs_resolver is None or outputs_resolver is None:
        raise ValueError(f"{fn!r} is not a @pipeline_step-decorated function")
    return StepDeclaration(
        inputs=tuple(inputs_resolver(*args, **kwargs)),
        outputs=tuple(outputs_resolver(*args, **kwargs)),
    )


def _missing_input_message(step_name: str, missing: Path) -> str:
    """Build the human-readable error for a missing declared input.

    Includes the producer hint when one is registered for the path.
    """
    base = f"[{step_name}] missing input: {missing}"
    producer = _lookup_producer(missing)
    if producer is not None:
        base += f"\n  produce it with: {producer}"
    return base


def _lookup_producer(path: Path) -> str | None:
    """Return the registered producer command for ``path``, or None."""
    s = str(path)
    for pattern, command in _PRODUCER_REGISTRY:
        if pattern in s:
            return command
    return None
