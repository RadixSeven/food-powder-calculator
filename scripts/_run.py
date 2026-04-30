"""Per-invocation run records under ``data/runs/<run_id>/manifest.yaml``.

Each top-level pipeline command (``group_pipeline.py``,
``extract_yaml.py``, …) wraps its main() in :func:`open_run`. While the
context is active, every :func:`@pipeline_step` invocation appends a
``StepRecord`` and every :func:`_claude.call` appends a ``CallRecord``.
On exit, the run writes ``manifest.yaml`` capturing:

* run identity: ``run_id``, ``git_sha``, ``git_dirty``, ``argv``,
  ``started_at`` / ``ended_at``;
* env fingerprint: ``python_version``, models used;
* per-step rows: name, started/ended, inputs (with sha256), outputs
  (with sha256);
* per-call rows: SHA, cached?, elapsed.

The manifest is what the provenance tool (next commit) queries to
answer "what run produced ``data/extracted_yaml/<gid>.yaml``?".

Run id format: ``<UTC-iso-second>-<short-git-sha>`` so runs are
sortable and disambiguated across machines / git states. A trailing
counter (``-1``, ``-2``) is appended if two runs collide on the
exact same second + sha.

The ``current_run`` contextvar lets ``@pipeline_step`` and
``_claude.call`` append into the active run without explicit
plumbing. When no run is open (tests, ad-hoc invocations) the
hooks are no-ops.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import platform
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "data" / "runs"


@dataclass(frozen=True)
class StepRecord:
    """One ``@pipeline_step`` invocation within a run.

    ``inputs`` / ``outputs`` are lists of (path, sha256) — the SHAs are
    captured at the time the step ran, so a later mutation of a file
    doesn't retroactively change a recorded input.
    """

    name: str
    started_at: str
    ended_at: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CallRecord:
    """One ``_claude.call`` invocation within a run."""

    sha: str
    cached: bool
    elapsed_seconds: float


@dataclass
class Run:
    """In-memory state for one open run, serialized to manifest.yaml on close."""

    run_id: str
    git_sha: str
    git_dirty: bool
    argv: tuple[str, ...]
    started_at: str
    runs_dir: Path
    ended_at: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    models_used: set[str] = field(default_factory=set)

    @property
    def directory(self) -> Path:
        """Absolute directory where this run's manifest will be written."""
        return self.runs_dir / self.run_id

    def record_step(self, record: StepRecord) -> None:
        """Append a step record (called by @pipeline_step on completion)."""
        self.steps.append(record)

    def record_call(
        self, *, sha: str, cached: bool, elapsed_seconds: float, model: str
    ) -> None:
        """Append an LLM-call record (called by _claude.call)."""
        self.calls.append(
            CallRecord(sha=sha, cached=cached, elapsed_seconds=elapsed_seconds)
        )
        self.models_used.add(model)

    def write_manifest(self) -> Path:
        """Serialize the run as ``<directory>/manifest.yaml``.

        Returns the path written. ``ended_at`` must already be set
        before calling — :func:`open_run` does this on exit.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "argv": list(self.argv),
            "env": {
                "python_version": platform.python_version(),
                "models_used": sorted(self.models_used),
            },
            "steps": [
                {
                    "name": s.name,
                    "started_at": s.started_at,
                    "ended_at": s.ended_at,
                    "inputs": [{"path": p, "sha256": h} for p, h in s.inputs],
                    "outputs": [{"path": p, "sha256": h} for p, h in s.outputs],
                }
                for s in self.steps
            ],
            "calls": [
                {
                    "sha": c.sha,
                    "cached": c.cached,
                    "elapsed_seconds": round(c.elapsed_seconds, 3),
                }
                for c in self.calls
            ],
        }
        manifest_path = self.directory / "manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        )
        return manifest_path


# ContextVar so concurrent (e.g. test) invocations don't leak run state
# across each other. Default ``None`` = no run open; @pipeline_step and
# _claude.call hooks become no-ops.
current_run: contextvars.ContextVar[Run | None] = contextvars.ContextVar(
    "current_run", default=None
)


@contextlib.contextmanager
def open_run(
    *,
    argv: Sequence[str] | None = None,
    runs_dir: Path = RUNS_DIR,
    now: datetime | None = None,
) -> Iterator[Run]:
    """Open a run, set it as :data:`current_run`, and write the manifest on exit.

    Use as ``with open_run(argv=sys.argv): main()``. The manifest is
    always written, even if the contained code raises — so a partial
    run is still auditable. The exception then propagates after the
    manifest write.

    ``now`` is a hook for tests so they can pin the run id; production
    callers leave it at the default (current wall-clock time).
    """
    run_argv = tuple(argv) if argv is not None else tuple(sys.argv)
    started = now if now is not None else datetime.now().astimezone()
    run_id = _make_run_id(started, _git_short_sha(), runs_dir)
    run = Run(
        run_id=run_id,
        git_sha=_git_sha() or "",
        git_dirty=_git_is_dirty(),
        argv=run_argv,
        started_at=started.isoformat(timespec="seconds"),
        runs_dir=runs_dir,
    )
    token = current_run.set(run)
    try:
        yield run
    finally:
        ended_dt = now if now is not None else datetime.now().astimezone()
        run.ended_at = ended_dt.isoformat(timespec="seconds")
        run.write_manifest()
        current_run.reset(token)


def sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 of ``path``'s bytes, or '' if the file is gone.

    Used to fingerprint inputs and outputs at the moment a step ran.
    A missing file (somehow deleted between step completion and SHA
    computation) yields an empty string rather than raising — the
    record still says the path was an input/output, just unverified.
    """
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_run_id(now: datetime, git_short: str, runs_dir: Path) -> str:
    """Build a unique run_id; append a counter if a same-second sibling exists.

    ``now`` is used in whatever timezone the caller supplied — the
    timestamp formatting just renders what's there. The default
    caller (:func:`open_run`) uses local time via
    ``datetime.now().astimezone()``; tests pin a tz-aware datetime
    directly for deterministic output.
    """
    base = now.strftime("%Y-%m-%dT%H-%M-%S")
    short = git_short or "no-git"
    candidate = f"{base}-{short}"
    if not (runs_dir / candidate).exists():
        return candidate
    i = 1
    while (runs_dir / f"{candidate}-{i}").exists():
        i += 1
    return f"{candidate}-{i}"


def _git_sha() -> str | None:
    """Return the current git HEAD SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_short_sha() -> str:
    """Return a 7-char short SHA, or '' if git isn't available."""
    sha = _git_sha()
    return sha[:7] if sha else ""


def _git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):  # pragma: no cover
        return False
    return result.returncode == 0 and bool(result.stdout.strip())
