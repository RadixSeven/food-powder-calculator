"""Tests for scripts/backfill_llm_records.py."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from backfill_llm_records import backfill, main


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_backfill_writes_full_record_when_request_and_response_match(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    records_dir = tmp_path / "records"
    sha = "abc123"
    _write(
        cache_dir / "responses" / f"{sha}.json",
        {"text": "the answer", "request_sha": sha},
    )
    _write(
        cache_dir / "requests" / f"{sha}.json",
        {"prompt": "hi", "model": "haiku", "image_paths": []},
    )

    counts = backfill(
        cache_dir=cache_dir,
        records_dir=records_dir,
        now=datetime.fromisoformat("2026-04-30T12:00:00+00:00"),
    )

    assert counts == {
        "written": 1,
        "skipped_existing": 0,
        "partial": 0,
        "unparseable": 0,
    }
    record = json.loads((records_dir / f"{sha}.json").read_text())
    assert record["request_sha"] == sha
    assert record["response"]["text"] == "the answer"
    assert record["request"]["prompt"] == "hi"
    assert "from_legacy_cache" not in record
    assert record["first_observed_at"] == "2026-04-30T12:00:00+00:00"


def test_backfill_writes_partial_record_when_request_is_missing(
    tmp_path: Path,
) -> None:
    """An old cache entry without its request file becomes a partial
    record marked ``from_legacy_cache``. Replay still works (same SHA
    matches), but the original request inputs are unrecoverable.
    """
    cache_dir = tmp_path / "cache"
    records_dir = tmp_path / "records"
    sha = "legacy"
    _write(cache_dir / "responses" / f"{sha}.json", {"text": "old answer"})

    counts = backfill(cache_dir=cache_dir, records_dir=records_dir)

    assert counts["written"] == 1
    assert counts["partial"] == 1
    record = json.loads((records_dir / f"{sha}.json").read_text())
    assert record["request"] is None
    assert record["from_legacy_cache"] is True
    assert record["response"]["text"] == "old answer"


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    """A second run leaves existing records untouched — preserves the
    first-observed-at timestamp recorded on the first backfill.
    """
    cache_dir = tmp_path / "cache"
    records_dir = tmp_path / "records"
    sha = "abc"
    _write(
        cache_dir / "responses" / f"{sha}.json",
        {"text": "answer", "request_sha": sha},
    )

    first = backfill(
        cache_dir=cache_dir,
        records_dir=records_dir,
        now=datetime.fromisoformat("2026-04-29T10:00:00+00:00"),
    )
    second = backfill(
        cache_dir=cache_dir,
        records_dir=records_dir,
        now=datetime.fromisoformat("2026-04-30T12:00:00+00:00"),
    )

    assert first["written"] == 1
    assert second["written"] == 0
    assert second["skipped_existing"] == 1
    record = json.loads((records_dir / f"{sha}.json").read_text())
    # First-observed-at preserved from the first backfill — second run
    # didn't clobber it.
    assert record["first_observed_at"] == "2026-04-29T10:00:00+00:00"


def test_backfill_skips_unparseable_response(tmp_path: Path) -> None:
    """A torn cache file is skipped with a warning rather than crashing
    the backfill, so a single bad file can't block the rest.
    """
    cache_dir = tmp_path / "cache"
    records_dir = tmp_path / "records"
    response_dir = cache_dir / "responses"
    response_dir.mkdir(parents=True)
    (response_dir / "abc.json").write_text("{not valid")
    _write(
        response_dir / "def.json",
        {"text": "fine"},
    )

    counts = backfill(cache_dir=cache_dir, records_dir=records_dir)

    assert counts["written"] == 1
    assert counts["unparseable"] == 1
    # The good one was written.
    assert (records_dir / "def.json").exists()
    assert not (records_dir / "abc.json").exists()


def test_backfill_skips_response_with_non_string_text(tmp_path: Path) -> None:
    """A response file whose 'text' field isn't a string is treated as
    unparseable — the records contract is text → str.
    """
    cache_dir = tmp_path / "cache"
    records_dir = tmp_path / "records"
    _write(
        cache_dir / "responses" / "abc.json",
        {"text": 42},
    )

    counts = backfill(cache_dir=cache_dir, records_dir=records_dir)
    assert counts["written"] == 0
    assert counts["unparseable"] == 1


def test_backfill_skips_response_with_top_level_array(tmp_path: Path) -> None:
    """Top-level non-object response is unparseable."""
    cache_dir = tmp_path / "cache"
    records_dir = tmp_path / "records"
    response_dir = cache_dir / "responses"
    response_dir.mkdir(parents=True)
    (response_dir / "abc.json").write_text(json.dumps([1, 2, 3]))

    counts = backfill(cache_dir=cache_dir, records_dir=records_dir)
    assert counts["unparseable"] == 1


def test_backfill_treats_unparseable_request_as_missing(tmp_path: Path) -> None:
    """A torn request file falls back to the 'request missing' code path
    rather than crashing — a torn request shouldn't lose the response.
    """
    cache_dir = tmp_path / "cache"
    records_dir = tmp_path / "records"
    sha = "abc"
    _write(
        cache_dir / "responses" / f"{sha}.json",
        {"text": "answer"},
    )
    (cache_dir / "requests" / f"{sha}.json").parent.mkdir(parents=True)
    (cache_dir / "requests" / f"{sha}.json").write_text("{torn")

    counts = backfill(cache_dir=cache_dir, records_dir=records_dir)
    assert counts["written"] == 1
    assert counts["partial"] == 1
    record = json.loads((records_dir / f"{sha}.json").read_text())
    assert record["request"] is None


def test_backfill_handles_missing_response_dir(tmp_path: Path) -> None:
    """A fresh repo with no cache yet returns zero counts cleanly."""
    counts = backfill(
        cache_dir=tmp_path / "missing", records_dir=tmp_path / "records"
    )
    assert counts == {
        "written": 0,
        "skipped_existing": 0,
        "partial": 0,
        "unparseable": 0,
    }


def test_main_prints_summary(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    """Smoke test the CLI: parses args, runs backfill, prints summary."""
    cache_dir = tmp_path / "cache"
    records_dir = tmp_path / "records"
    _write(cache_dir / "responses" / "abc.json", {"text": "ok"})

    import sys

    monkeypatch.setattr(  # type: ignore[attr-defined]
        sys,
        "argv",
        [
            "backfill_llm_records.py",
            "--cache-dir",
            str(cache_dir),
            "--records-dir",
            str(records_dir),
        ],
    )
    rc = main()
    assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "written=1" in captured.out
    assert "partial-from-legacy=1" in captured.out
