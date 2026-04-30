"""Per-batch metadata loader for ``data/raw_photos/<batch>/batch.yaml``.

Each batch directory holds the photos it contains plus a ``batch.yaml``
that records who collected them, when, and how they map to stores.
The store assignment is per-photo: a phone batch typically has a
filename cutoff between stores (the photographer walked from MOM to
CVS); an online-retailer batch typically has one photo per product
with the store baked into the rule.

The schema is deliberately small. Today's pipeline only needs the
store-for-photo lookup; later batches can grow new fields (URLs,
prices captured at fetch time, per-photo notes) without breaking the
loader.

Schema::

    name: <batch-name>            # required, must match the directory name
    captured_at: <YYYY-MM-DD>     # required, ISO date
    source: <free-form string>    # required, e.g. "phone-camera"
    notes: |                      # optional, free text for human review
      ...

    # First matching rule wins; rules are evaluated in source order.
    # `until` is a lexically-sorted filename cutoff, inclusive. Omit
    # it on the trailing rule to mean "everything else in the batch".
    stores:
      - { name: MOM, until: PXL_20260426_180743283.jpg }
      - { name: CVS }
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PHOTOS_DIR = REPO_ROOT / "data" / "raw_photos"
BATCH_FILENAME = "batch.yaml"


@dataclass(frozen=True)
class StoreRule:
    """One store assignment rule from a batch.yaml ``stores`` list."""

    name: str
    until: str | None  # inclusive filename cutoff; None = "everything else"


@dataclass(frozen=True)
class Batch:
    """A loaded ``batch.yaml`` plus the directory it came from."""

    name: str
    directory: Path
    captured_at: str
    source: str
    notes: str
    stores: tuple[StoreRule, ...]

    def store_for(self, photo_filename: str) -> str:
        """Return the store name for ``photo_filename`` per this batch's rules.

        The first rule whose ``until`` is ``None`` or whose
        ``until`` is greater-than-or-equal to the filename wins. If
        no rule matches (no trailing catch-all and the filename
        sorts past every cutoff), raise ``KeyError``.
        """
        for rule in self.stores:
            if rule.until is None:
                return rule.name
            if photo_filename <= rule.until:
                return rule.name
        raise KeyError(
            f"No store rule matched {photo_filename!r} in batch {self.name!r}; "
            f"rules: {[(r.name, r.until) for r in self.stores]}"
        )


def load_batch(batch_dir: Path) -> Batch:
    """Read ``<batch_dir>/batch.yaml`` and return a :class:`Batch`.

    Raises ``FileNotFoundError`` if the file isn't there, ``ValueError``
    if it's malformed (missing required keys, wrong shape), or
    ``yaml.YAMLError`` on a parse failure. Failing loud at the boundary
    is on purpose — silent fallbacks would mask drift between batches.
    """
    yaml_path = batch_dir / BATCH_FILENAME
    if not yaml_path.exists():
        raise FileNotFoundError(f"No {BATCH_FILENAME} at {yaml_path}")
    payload = yaml.safe_load(yaml_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{yaml_path}: top-level must be a mapping")
    name = payload.get("name")
    # PyYAML auto-parses bare ISO dates (``2026-04-26``) into
    # ``datetime.date``; accept either form and normalize to a string
    # so downstream code only sees one shape.
    raw_captured_at = payload.get("captured_at")
    if isinstance(raw_captured_at, (dt.date, dt.datetime)):
        captured_at: str | None = raw_captured_at.isoformat()
    elif isinstance(raw_captured_at, str):
        captured_at = raw_captured_at
    else:
        captured_at = None
    source = payload.get("source")
    notes_value = payload.get("notes", "")
    stores_value = payload.get("stores")
    if not isinstance(name, str):
        raise ValueError(f"{yaml_path}: 'name' must be a string")
    if captured_at is None:
        raise ValueError(
            f"{yaml_path}: 'captured_at' must be a YYYY-MM-DD string or date"
        )
    if not isinstance(source, str):
        raise ValueError(f"{yaml_path}: 'source' must be a string")
    if not isinstance(stores_value, list) or not stores_value:
        raise ValueError(
            f"{yaml_path}: 'stores' must be a non-empty list of rules"
        )
    rules: list[StoreRule] = []
    for i, raw_rule in enumerate(stores_value):
        if not isinstance(raw_rule, dict):
            raise ValueError(
                f"{yaml_path}: stores[{i}] must be a mapping, got {raw_rule!r}"
            )
        rule_name = raw_rule.get("name")
        rule_until = raw_rule.get("until")
        if not isinstance(rule_name, str):
            raise ValueError(f"{yaml_path}: stores[{i}].name must be a string")
        if rule_until is not None and not isinstance(rule_until, str):
            raise ValueError(
                f"{yaml_path}: stores[{i}].until must be a string or null"
            )
        rules.append(StoreRule(name=rule_name, until=rule_until))
    notes = notes_value if isinstance(notes_value, str) else ""
    return Batch(
        name=name,
        directory=batch_dir,
        captured_at=captured_at,
        source=source,
        notes=notes,
        stores=tuple(rules),
    )


def find_batch_for_photo(photo_path: Path) -> Batch:
    """Locate the batch a photo belongs to by walking up to its ``batch.yaml``.

    The photo's immediate parent directory must contain ``batch.yaml``.
    We don't recurse upward: if a photo isn't in a batch directory the
    layout is wrong, and a clear error is better than a silent guess.
    """
    parent = photo_path.parent
    if not (parent / BATCH_FILENAME).exists():
        raise FileNotFoundError(
            f"Photo {photo_path} is not in a batch directory "
            f"(no {BATCH_FILENAME} alongside it)"
        )
    return load_batch(parent)


def store_for_photo(photo_path: Path) -> str:
    """Return the store name for ``photo_path`` via its batch.yaml rules.

    Wrapper that combines :func:`find_batch_for_photo` with
    :meth:`Batch.store_for` — the common case where the caller has a
    photo path and just wants the store.
    """
    batch = find_batch_for_photo(photo_path)
    return batch.store_for(photo_path.name)
