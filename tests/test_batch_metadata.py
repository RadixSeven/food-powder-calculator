"""Tests for scripts/batch_metadata.py."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from batch_metadata import (
    Batch,
    StoreRule,
    find_batch_for_photo,
    load_batch,
    store_for_photo,
)


def _write_batch_yaml(batch_dir: Path, body: str) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "batch.yaml").write_text(dedent(body))


# ---------------------------------------------------------------------------
# load_batch
# ---------------------------------------------------------------------------


def test_load_batch_parses_minimal_well_formed_yaml(tmp_path: Path) -> None:
    batch_dir = tmp_path / "2026-04-26-shopping"
    _write_batch_yaml(
        batch_dir,
        """\
        name: 2026-04-26-shopping
        captured_at: 2026-04-26
        source: phone-camera
        notes: |
          208 photos.
        stores:
          - {name: MOM, until: PXL_20260426_180743283.jpg}
          - {name: CVS}
        """,
    )
    batch = load_batch(batch_dir)
    assert batch.name == "2026-04-26-shopping"
    assert batch.captured_at == "2026-04-26"
    assert batch.source == "phone-camera"
    assert "208 photos" in batch.notes
    assert batch.stores == (
        StoreRule(name="MOM", until="PXL_20260426_180743283.jpg"),
        StoreRule(name="CVS", until=None),
    )
    assert batch.directory == batch_dir


def test_load_batch_raises_for_missing_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_batch(tmp_path / "no-such-batch")


def test_load_batch_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    batch_dir = tmp_path / "b"
    batch_dir.mkdir()
    (batch_dir / "batch.yaml").write_text("- just a list\n- of things\n")
    with pytest.raises(ValueError, match="top-level must be a mapping"):
        load_batch(batch_dir)


def test_load_batch_rejects_missing_required_fields(tmp_path: Path) -> None:
    """Required fields are required — silent defaults would mask drift
    between batches.
    """
    cases = [
        (
            "missing name",
            "captured_at: 2026-04-26\nsource: x\nstores: [{name: A}]",
        ),
        ("missing captured_at", "name: x\nsource: y\nstores: [{name: A}]"),
        (
            "missing source",
            "name: x\ncaptured_at: 2026-04-26\nstores: [{name: A}]",
        ),
        ("missing stores", "name: x\ncaptured_at: 2026-04-26\nsource: y"),
        (
            "empty stores",
            "name: x\ncaptured_at: 2026-04-26\nsource: y\nstores: []",
        ),
    ]
    for label, body in cases:
        batch_dir = tmp_path / label.replace(" ", "_")
        batch_dir.mkdir()
        (batch_dir / "batch.yaml").write_text(body)
        with pytest.raises(ValueError):
            load_batch(batch_dir)


def test_load_batch_rejects_non_string_until(tmp_path: Path) -> None:
    batch_dir = tmp_path / "b"
    _write_batch_yaml(
        batch_dir,
        """\
        name: b
        captured_at: 2026-04-26
        source: y
        stores:
          - {name: MOM, until: 42}
        """,
    )
    with pytest.raises(ValueError, match="stores\\[0\\].until"):
        load_batch(batch_dir)


def test_load_batch_rejects_non_string_rule_name(tmp_path: Path) -> None:
    batch_dir = tmp_path / "b"
    _write_batch_yaml(
        batch_dir,
        """\
        name: b
        captured_at: 2026-04-26
        source: y
        stores:
          - {name: 42}
        """,
    )
    with pytest.raises(ValueError, match="stores\\[0\\].name"):
        load_batch(batch_dir)


def test_load_batch_rejects_non_mapping_rule(tmp_path: Path) -> None:
    batch_dir = tmp_path / "b"
    _write_batch_yaml(
        batch_dir,
        """\
        name: b
        captured_at: 2026-04-26
        source: y
        stores:
          - "MOM"
        """,
    )
    with pytest.raises(ValueError, match="stores\\[0\\] must be a mapping"):
        load_batch(batch_dir)


def test_load_batch_accepts_quoted_string_captured_at(tmp_path: Path) -> None:
    """PyYAML normally parses a bare ``2026-04-26`` as ``datetime.date``,
    but a quoted ``"2026-04-26"`` is a string. Both forms must work.
    """
    batch_dir = tmp_path / "b"
    _write_batch_yaml(
        batch_dir,
        """\
        name: b
        captured_at: "2026-04-26"
        source: y
        stores:
          - {name: A}
        """,
    )
    batch = load_batch(batch_dir)
    assert batch.captured_at == "2026-04-26"


def test_load_batch_rejects_non_string_source(tmp_path: Path) -> None:
    batch_dir = tmp_path / "b"
    _write_batch_yaml(
        batch_dir,
        """\
        name: b
        captured_at: 2026-04-26
        source: 42
        stores:
          - {name: A}
        """,
    )
    with pytest.raises(ValueError, match="'source' must be a string"):
        load_batch(batch_dir)


def test_load_batch_treats_non_string_notes_as_empty(tmp_path: Path) -> None:
    """A misshapen notes field shouldn't fail the load — notes are
    user-facing, the rest of the schema isn't sensitive to it.
    """
    batch_dir = tmp_path / "b"
    _write_batch_yaml(
        batch_dir,
        """\
        name: b
        captured_at: 2026-04-26
        source: y
        notes: 42
        stores:
          - {name: A}
        """,
    )
    batch = load_batch(batch_dir)
    assert batch.notes == ""


# ---------------------------------------------------------------------------
# store_for / find_batch_for_photo / store_for_photo
# ---------------------------------------------------------------------------


def test_batch_store_for_returns_first_matching_rule(tmp_path: Path) -> None:
    batch_dir = tmp_path / "b"
    _write_batch_yaml(
        batch_dir,
        """\
        name: b
        captured_at: 2026-04-26
        source: y
        stores:
          - {name: MOM, until: PXL_20260426_180743283.jpg}
          - {name: CVS}
        """,
    )
    batch = load_batch(batch_dir)
    # Just before cutoff → MOM.
    assert batch.store_for("PXL_20260426_180000000.jpg") == "MOM"
    # Exactly at cutoff (inclusive) → MOM.
    assert batch.store_for("PXL_20260426_180743283.jpg") == "MOM"
    # Past cutoff → CVS.
    assert batch.store_for("PXL_20260426_180942709.MP.jpg") == "CVS"


def test_batch_store_for_raises_when_no_rule_matches(tmp_path: Path) -> None:
    """If every rule has an `until` and the photo sorts past all of
    them, we raise — silent default would let typo'd batches drift.
    """
    batch_dir = tmp_path / "b"
    _write_batch_yaml(
        batch_dir,
        """\
        name: b
        captured_at: 2026-04-26
        source: y
        stores:
          - {name: A, until: PXL_a.jpg}
          - {name: B, until: PXL_b.jpg}
        """,
    )
    batch = load_batch(batch_dir)
    with pytest.raises(KeyError):
        batch.store_for("PXL_z.jpg")


def test_find_batch_for_photo_walks_to_parent_dir(tmp_path: Path) -> None:
    batch_dir = tmp_path / "2026-04-26-shopping"
    _write_batch_yaml(
        batch_dir,
        """\
        name: 2026-04-26-shopping
        captured_at: 2026-04-26
        source: phone-camera
        stores:
          - {name: MOM}
        """,
    )
    photo = batch_dir / "PXL_x.jpg"
    photo.write_bytes(b"fake")
    batch = find_batch_for_photo(photo)
    assert batch.name == "2026-04-26-shopping"


def test_find_batch_for_photo_raises_when_photo_outside_batch(
    tmp_path: Path,
) -> None:
    """No upward recursion: a photo whose parent has no batch.yaml is a
    layout error, not a we'll-search-around-and-guess situation.
    """
    bare = tmp_path / "loose-photo.jpg"
    bare.write_bytes(b"fake")
    with pytest.raises(FileNotFoundError):
        find_batch_for_photo(bare)


def test_store_for_photo_combines_locator_and_store_for(tmp_path: Path) -> None:
    batch_dir = tmp_path / "b"
    _write_batch_yaml(
        batch_dir,
        """\
        name: b
        captured_at: 2026-04-26
        source: y
        stores:
          - {name: MOM, until: PXL_20260426_180743283.jpg}
          - {name: CVS}
        """,
    )
    early = batch_dir / "PXL_20260426_165737642.jpg"
    early.write_bytes(b"fake")
    late = batch_dir / "PXL_20260426_181135293.jpg"
    late.write_bytes(b"fake")
    assert store_for_photo(early) == "MOM"
    assert store_for_photo(late) == "CVS"


def test_batch_dataclass_is_frozen() -> None:
    """Frozen so a Batch is safe to memoize / pass around without copying."""
    rule = StoreRule(name="A", until=None)
    batch = Batch(
        name="b",
        directory=Path("."),
        captured_at="2026-04-26",
        source="x",
        notes="",
        stores=(rule,),
    )
    with pytest.raises(Exception):
        batch.name = "other"  # type: ignore[misc]
