"""Tests for scripts/find_legible_size.py.

The Claude calls are mocked everywhere; image resizing uses real Pillow on
small synthetic images.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from _claude import ClaudeStructuredOutputError
from find_legible_size import (
    EXTRACTION_JSON_SCHEMA,
    EXTRACTION_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    ExtractedPayload,
    binary_search_min_size,
    extract_payload,
    matches_reference,
    select_search_model,
    _extract_json_object,
    _levenshtein,
    _levenshtein_ratio,
    _parse_payload,
)
from PIL import Image


# ---------- pure-function tests ------------------------------------------


def test_levenshtein_zero_for_equal_strings() -> None:
    assert _levenshtein("hello", "hello") == 0


def test_levenshtein_handles_empty() -> None:
    assert _levenshtein("", "abc") == 3
    assert _levenshtein("abc", "") == 3
    assert _levenshtein("", "") == 0


def test_levenshtein_basic_substitution() -> None:
    assert _levenshtein("kitten", "sitting") == 3


def test_levenshtein_ratio_perfect() -> None:
    assert _levenshtein_ratio("hello", "hello") == 1.0


def test_levenshtein_ratio_empty() -> None:
    assert _levenshtein_ratio("", "") == 1.0


def test_levenshtein_ratio_low() -> None:
    assert _levenshtein_ratio("hello", "world") < 0.5


def test_extract_json_object_from_fenced_block() -> None:
    raw = 'here is the answer:\n```json\n{"text": "hi", "barcodes": []}\n```\nthanks'
    assert '"text"' in _extract_json_object(raw)


def test_extract_json_object_from_bare_object() -> None:
    raw = 'prose before {"text": "hi"} prose after'
    assert _extract_json_object(raw) == '{"text": "hi"}'


def test_extract_json_object_raises_on_missing_object() -> None:
    with pytest.raises(ValueError, match="No JSON object"):
        _extract_json_object("just plain prose with no braces")


def test_parse_payload_normalizes_text_and_barcodes() -> None:
    payload = _parse_payload('{"text": "Hello World", "barcodes": ["12345"]}')
    assert payload.text == "hello world"
    assert payload.barcodes == ("12345",)


def test_matches_reference_text_below_threshold() -> None:
    ref = ExtractedPayload(text="abcdefghij", barcodes=())
    cand = ExtractedPayload(text="zzzzzzzzzz", barcodes=())
    assert matches_reference(ref, cand) is False


def test_matches_reference_text_above_threshold_no_barcodes() -> None:
    ref = ExtractedPayload(text="abcdefghij", barcodes=())
    cand = ExtractedPayload(text="abcdefghij", barcodes=())
    assert matches_reference(ref, cand) is True


def test_matches_reference_default_ignores_barcodes() -> None:
    """Default mode passes any text-similar candidate, regardless of barcodes
    — barcodes degrade before the surrounding text under downsampling."""
    ref = ExtractedPayload(text="hello", barcodes=("123", "456"))
    has_one = ExtractedPayload(text="hello", barcodes=("123",))
    has_none = ExtractedPayload(text="hello", barcodes=())
    assert matches_reference(ref, has_one) is True
    assert matches_reference(ref, has_none) is True


def test_matches_reference_strict_barcode_mode_requires_every_one() -> None:
    """Strict mode is opt-in for callers that genuinely need barcode parity."""
    ref = ExtractedPayload(text="hello", barcodes=("123", "456"))
    has_one = ExtractedPayload(text="hello", barcodes=("123",))
    has_both = ExtractedPayload(text="hello", barcodes=("123", "456"))
    assert matches_reference(ref, has_one, require_barcodes=True) is False
    assert matches_reference(ref, has_both, require_barcodes=True) is True


def test_matches_reference_extra_candidate_barcodes_are_fine() -> None:
    ref = ExtractedPayload(text="hello", barcodes=("123",))
    cand = ExtractedPayload(text="hello", barcodes=("123", "999"))
    assert matches_reference(ref, cand) is True


# ---------- mocked-Claude tests ------------------------------------------


def _solid_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (200, 50, 50)).save(path, "JPEG")


def test_extract_payload_passes_through_to_claude(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (4000, 2000))

    fake_response = MagicMock()
    fake_response.text = '{"text": "ok", "barcodes": ["123"]}'
    with patch("find_legible_size.RESIZE_CACHE_DIR", tmp_path / "resized"):
        with patch(
            "find_legible_size.call", return_value=fake_response
        ) as mock_call:
            payload = extract_payload(src, "haiku", longest_side=512)

    assert payload.text == "ok"
    assert payload.barcodes == ("123",)
    request = mock_call.call_args.args[0]
    assert request.model == "haiku"
    assert request.system_prompt == EXTRACTION_SYSTEM_PROMPT
    assert request.json_schema == EXTRACTION_JSON_SCHEMA
    assert request.prompt == EXTRACTION_PROMPT


def test_extract_payload_uses_full_resolution_when_no_longest_side(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (1000, 500))

    fake_response = MagicMock()
    fake_response.text = '{"text": "ok", "barcodes": []}'
    with patch(
        "find_legible_size.call", return_value=fake_response
    ) as mock_call:
        extract_payload(src, "opus")

    request = mock_call.call_args.args[0]
    # Image path should be the original, not a resized cache path.
    assert request.image_paths == (src,)


def test_select_search_model_picks_first_match(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (1000, 500))

    reference = ExtractedPayload(text="hello world", barcodes=("123",))

    def fake_extract(
        _path: Path, model: str, longest_side: int | None = None
    ) -> ExtractedPayload:
        # Haiku gets it right.
        return (
            ExtractedPayload(text="hello world", barcodes=("123",))
            if model == "haiku"
            else ExtractedPayload(text="zzz", barcodes=())
        )

    with patch("find_legible_size.extract_payload", side_effect=fake_extract):
        model, payload = select_search_model(src, reference)

    assert model == "haiku"
    assert payload.barcodes == ("123",)


def test_select_search_model_falls_back_to_sonnet(tmp_path: Path) -> None:
    """Haiku fails, sonnet matches → sonnet is the search model."""
    src = tmp_path / "src.jpg"
    _solid_image(src, (1000, 500))

    reference = ExtractedPayload(text="hello world", barcodes=())

    def fake_extract(
        _path: Path, model: str, longest_side: int | None = None
    ) -> ExtractedPayload:
        if model == "haiku":
            return ExtractedPayload(text="zzz", barcodes=())
        return ExtractedPayload(text="hello world", barcodes=())

    with patch("find_legible_size.extract_payload", side_effect=fake_extract):
        model, _payload = select_search_model(src, reference)

    assert model == "sonnet"


def test_select_search_model_falls_back_to_last_when_none_match(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (1000, 500))

    reference = ExtractedPayload(text="hello world", barcodes=("123",))
    bad = ExtractedPayload(text="zzzz", barcodes=())

    with patch("find_legible_size.extract_payload", return_value=bad):
        model, payload = select_search_model(src, reference)

    assert model == "opus"  # last in the default tuple
    assert payload == bad


def test_binary_search_returns_smallest_passing_size(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (2048, 1024))

    reference = ExtractedPayload(text="needle", barcodes=())
    threshold = 512  # match if longest_side >= 512

    def fake_extract(
        _path: Path, _model: str, longest_side: int | None = None
    ) -> ExtractedPayload:
        assert longest_side is not None
        size = longest_side
        return (
            reference
            if size >= threshold
            else ExtractedPayload(text="garbled", barcodes=())
        )

    with patch("find_legible_size.RESIZE_CACHE_DIR", tmp_path / "resized"):
        with patch(
            "find_legible_size.extract_payload", side_effect=fake_extract
        ):
            result = binary_search_min_size(
                src, "haiku", reference, photo_id="p"
            )

    assert result.min_legible_size == threshold
    assert all(p.photo_id == "p" for p in result.probes)
    # At least one probe at exactly the threshold returned matched=True.
    matched_at_threshold = [
        p for p in result.probes if p.size == threshold and p.matched
    ]
    assert matched_at_threshold


def test_binary_search_handles_image_smaller_than_lower_bound(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.jpg"
    _solid_image(src, (40, 40))  # below default lower_bound=64
    reference = ExtractedPayload(text="needle", barcodes=())

    result = binary_search_min_size(src, "haiku", reference, photo_id="p")
    assert result.min_legible_size == 40
    assert len(result.probes) == 1


def test_binary_search_skips_lucky_pass_when_a_larger_probe_failed(
    tmp_path: Path,
) -> None:
    """Implements the non-monotonic correction post-search.

    The fake model passes at small probed sizes but fails at a larger size;
    the algorithm must skip the small (lucky) passes and return the upper.
    """
    src = tmp_path / "src.jpg"
    _solid_image(src, (2048, 1024))

    reference = ExtractedPayload(text="needle", barcodes=())

    def fake_extract(
        _path: Path, _model: str, longest_side: int | None = None
    ) -> ExtractedPayload:
        assert longest_side is not None
        # Fail at sizes in (600, 1500], pass at > 1500. Pure binary search
        # converges in (1500, 2048]; the post-search filter must NOT promote
        # any earlier "passing" probe back into the answer.
        return (
            reference
            if longest_side > 1500
            else ExtractedPayload(text="g", barcodes=())
        )

    with patch("find_legible_size.RESIZE_CACHE_DIR", tmp_path / "resized"):
        with patch(
            "find_legible_size.extract_payload", side_effect=fake_extract
        ):
            result = binary_search_min_size(
                src, "haiku", reference, photo_id="p"
            )

    # We must NOT trust a pass <= 600 when a mid in (600, 1500] failed.
    # Either the algorithm settled on a size > 600, or it gave up on the
    # full image upper bound.
    assert result.min_legible_size > 600


def test_binary_search_treats_structured_output_failure_as_no_match(
    tmp_path: Path,
) -> None:
    """At small sizes the model can fail to produce valid JSON
    (ClaudeStructuredOutputError). Binary search must treat that as a
    ``matched=False`` probe and keep searching, not crash."""
    src = tmp_path / "src.jpg"
    _solid_image(src, (2048, 1024))

    reference = ExtractedPayload(text="needle", barcodes=())

    def fake_extract(
        _path: Path, _model: str, longest_side: int | None = None
    ) -> ExtractedPayload:
        assert longest_side is not None
        # Match at >= 1024; under 1024 the "model" raises structured-output
        # exhaustion (mimicking the live failure mode at low resolutions).
        if longest_side < 1024:
            raise ClaudeStructuredOutputError(
                returncode=1,
                stdout="",
                stderr="",
                cmd=[],
                failure_message="schema-invalid output after 5 retries",
            )
        return reference

    with patch("find_legible_size.RESIZE_CACHE_DIR", tmp_path / "resized"):
        with patch(
            "find_legible_size.extract_payload", side_effect=fake_extract
        ):
            result = binary_search_min_size(
                src, "haiku", reference, photo_id="p"
            )

    # The search should still converge — small-size probes are recorded
    # as no-match instead of crashing.
    assert result.min_legible_size >= 1024
    # And the failed probes are present in the record (matched=False).
    assert any(not p.matched for p in result.probes)


def test_binary_search_returns_image_upper_when_no_probe_passes(
    tmp_path: Path,
) -> None:
    """If no probed size passes, return the image's full resolution."""
    src = tmp_path / "src.jpg"
    _solid_image(src, (2048, 1024))

    # Every probe fails (doesn't matter what reference is).
    fail = ExtractedPayload(text="garbled", barcodes=())

    with patch("find_legible_size.RESIZE_CACHE_DIR", tmp_path / "resized"):
        with patch("find_legible_size.extract_payload", return_value=fail):
            result = binary_search_min_size(
                src,
                "haiku",
                ExtractedPayload(text="needle", barcodes=()),
                photo_id="p",
            )

    assert result.min_legible_size == 2048
    assert all(p.matched is False for p in result.probes)
