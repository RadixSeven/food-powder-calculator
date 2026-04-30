"""Pytest fixtures shared across the test suite.

The autouse ``isolate_llm_records`` fixture redirects
:data:`_claude.LLM_RECORDS_DIR` at a per-test temp directory. Without
this, any test that ends up exercising :func:`_claude.call` (directly,
or via a higher-level pipeline test that mocks the subprocess) would
write into the real ``data/llm_records/`` and pollute the tracked
durable-records store. The cache (:data:`_claude.CACHE_DIR`) is already
patched explicitly by the tests that care about its contents; records
are autouse because most tests don't think about them at all and we
want the wrong default to be the safe one.

The fixture skips silently when :mod:`_claude` isn't importable from
the current test's sys.path (the ``src``-only tests don't add the
``scripts`` directory, so they can't reach the module — but they also
can't trigger LLM calls, so the lack of redirection is harmless).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_llm_records(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path | None:
    """Redirect ``_claude.LLM_RECORDS_DIR`` to a per-test temp path.

    Returned for the rare test that needs to assert on the records
    directory's contents — most tests don't reference it at all.
    Returns ``None`` for tests where :mod:`_claude` isn't reachable
    (tests under ``src``-only sys.path don't import or exercise it).
    """
    try:
        importlib.import_module("_claude")
    except ImportError:
        return None
    records_dir = tmp_path_factory.mktemp("llm_records")
    monkeypatch.setattr("_claude.LLM_RECORDS_DIR", records_dir)
    return records_dir


@pytest.fixture(autouse=True)
def isolate_runs_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path | None:
    """Redirect ``_run.RUNS_DIR`` to a per-test temp path.

    Same rationale as :func:`isolate_llm_records` — tests that
    happen to call ``open_run`` without explicitly passing
    ``runs_dir=`` would otherwise write into the real
    ``data/runs/`` audit trail.
    """
    try:
        importlib.import_module("_run")
    except ImportError:
        return None
    runs_dir = tmp_path_factory.mktemp("runs")
    monkeypatch.setattr("_run.RUNS_DIR", runs_dir)
    return runs_dir
