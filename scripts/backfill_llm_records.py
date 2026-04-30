"""Promote existing ``data/cache/responses/`` entries to durable records.

The cache predates the durable-records layer (:data:`_claude.LLM_RECORDS_DIR`).
This one-shot script walks the cache and writes the matching durable
record for every response present, so the existing ~1800 cached
responses survive a future ``rm -rf data/cache``.

Three response shapes to handle:

* response present + request present  → full durable record.
* response present + request absent   → partial record; ``request`` is
  ``null`` and a ``from_legacy_cache: true`` marker explains why. The
  record is still useful for replay (it satisfies a same-SHA call), but
  the original request inputs are gone — we have a hash, and SHA is
  one-way.
* response file unparseable           → skipped with a stderr warning.

Idempotent: a record that already exists is left untouched so a
re-run doesn't perturb the ``first_observed_at`` timestamps recorded
on the first backfill.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from _claude import (
    CACHE_DIR,
    LLM_RECORDS_DIR,
    REQUEST_CACHE_SUBDIR,
    RESPONSE_CACHE_SUBDIR,
    _atomic_write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def backfill(
    *,
    cache_dir: Path = CACHE_DIR,
    records_dir: Path = LLM_RECORDS_DIR,
    now: datetime | None = None,
) -> dict[str, int]:
    """Walk the cache and write a durable record for every response.

    Returns a dict of counts keyed by ``"written"``, ``"skipped_existing"``,
    ``"partial"`` (record written without request), and ``"unparseable"``
    so callers (tests + the CLI) can render a summary.
    """
    response_dir = cache_dir / RESPONSE_CACHE_SUBDIR
    request_dir = cache_dir / REQUEST_CACHE_SUBDIR
    if not response_dir.exists():
        return {
            "written": 0,
            "skipped_existing": 0,
            "partial": 0,
            "unparseable": 0,
        }
    timestamp = (now or datetime.now().astimezone()).isoformat(
        timespec="seconds"
    )
    counts = {
        "written": 0,
        "skipped_existing": 0,
        "partial": 0,
        "unparseable": 0,
    }
    for response_path in sorted(response_dir.glob("*.json")):
        sha = response_path.stem
        record_path = records_dir / f"{sha}.json"
        if record_path.exists():
            counts["skipped_existing"] += 1
            continue
        try:
            response_payload = json.loads(response_path.read_text())
        except (json.JSONDecodeError, ValueError):
            print(
                f"[backfill] {sha}: unparseable response file, skipping",
                file=sys.stderr,
            )
            counts["unparseable"] += 1
            continue
        text = (
            response_payload.get("text")
            if isinstance(response_payload, dict)
            else None
        )
        if not isinstance(text, str):
            print(
                f"[backfill] {sha}: response has no string 'text', skipping",
                file=sys.stderr,
            )
            counts["unparseable"] += 1
            continue
        request_path = request_dir / f"{sha}.json"
        request_payload: object | None = None
        if request_path.exists():
            try:
                request_payload = json.loads(request_path.read_text())
            except (json.JSONDecodeError, ValueError):
                request_payload = None
        record: dict[str, object] = {
            "request_sha": sha,
            "request": request_payload,
            "response": {"text": text},
            "first_observed_at": timestamp,
        }
        if request_payload is None:
            record["from_legacy_cache"] = True
            counts["partial"] += 1
        _atomic_write_json(record_path, record)
        counts["written"] += 1
    return counts


def main() -> int:
    """Command-line entry: print a one-line summary of the backfill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Cache directory to read from (data/cache/ by default).",
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=LLM_RECORDS_DIR,
        help="Records directory to write to (data/llm_records/ by default).",
    )
    args = parser.parse_args()
    counts = backfill(cache_dir=args.cache_dir, records_dir=args.records_dir)
    print(
        f"[backfill] written={counts['written']} "
        f"(of which partial-from-legacy={counts['partial']}) "
        f"skipped_existing={counts['skipped_existing']} "
        f"unparseable={counts['unparseable']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
