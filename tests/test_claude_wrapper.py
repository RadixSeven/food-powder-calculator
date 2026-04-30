"""Tests for scripts/_claude.py — the cached `claude -p` subprocess wrapper.

The actual `claude` subprocess is mocked everywhere; these tests run with
no network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from _claude import (
    MAX_RATE_LIMIT_RETRIES,
    MIN_RATE_LIMIT_SLEEP_SECONDS,
    RATE_LIMIT_FALLBACK_SLEEP_SECONDS,
    ClaudeRateLimitError,
    ClaudeRequest,
    ClaudeResponse,
    ClaudeStructuredOutputError,
    ClaudeSubprocessError,
    _all_dirs,
    _compose_stdin_prompt,
    _compute_rate_limit_sleep_seconds,
    _detect_rate_limit,
    _detect_structured_output_failure,
    _file_sha,
    _parse_reset_time,
    _request_sha,
    call,
)


def _png_bytes(rgb: tuple[int, int, int]) -> bytes:
    """Smallest possible PNG file with the given RGB color (1×1)."""
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), rgb).save(buf, format="PNG")
    return buf.getvalue()


def _make_image(tmp_path: Path, name: str, rgb: tuple[int, int, int]) -> Path:
    p = tmp_path / name
    p.write_bytes(_png_bytes(rgb))
    return p


def test_request_sha_is_stable_for_identical_inputs(tmp_path: Path) -> None:
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r1 = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    r2 = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    assert _request_sha(r1) == _request_sha(r2)


def test_request_sha_changes_when_prompt_changes() -> None:
    r1 = ClaudeRequest(prompt="hi", model="haiku")
    r2 = ClaudeRequest(prompt="bye", model="haiku")
    assert _request_sha(r1) != _request_sha(r2)


def test_request_sha_changes_when_model_changes() -> None:
    r1 = ClaudeRequest(prompt="hi", model="haiku")
    r2 = ClaudeRequest(prompt="hi", model="opus")
    assert _request_sha(r1) != _request_sha(r2)


def test_request_sha_changes_when_image_bytes_change(tmp_path: Path) -> None:
    img1 = _make_image(tmp_path, "a.png", (255, 0, 0))
    img2 = _make_image(tmp_path, "b.png", (0, 255, 0))
    r1 = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img1,))
    r2 = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img2,))
    assert _request_sha(r1) != _request_sha(r2)


def test_request_sha_changes_when_cache_version_bumps() -> None:
    r1 = ClaudeRequest(prompt="hi", model="haiku", cache_version=1)
    r2 = ClaudeRequest(prompt="hi", model="haiku", cache_version=2)
    assert _request_sha(r1) != _request_sha(r2)


def test_request_sha_changes_when_system_prompt_changes() -> None:
    r1 = ClaudeRequest(prompt="hi", model="haiku", system_prompt="be brief")
    r2 = ClaudeRequest(prompt="hi", model="haiku", system_prompt="be verbose")
    assert _request_sha(r1) != _request_sha(r2)


def test_request_sha_changes_when_json_schema_changes() -> None:
    r1 = ClaudeRequest(prompt="hi", model="haiku", json_schema='{"a": 1}')
    r2 = ClaudeRequest(prompt="hi", model="haiku", json_schema='{"b": 2}')
    assert _request_sha(r1) != _request_sha(r2)


def test_compose_stdin_prompt_no_images() -> None:
    r = ClaudeRequest(prompt="just text", model="haiku")
    assert _compose_stdin_prompt(r) == "just text"


def test_compose_stdin_prompt_with_images_lists_paths(tmp_path: Path) -> None:
    img1 = _make_image(tmp_path, "a.png", (255, 0, 0))
    img2 = _make_image(tmp_path, "b.png", (0, 255, 0))
    r = ClaudeRequest(
        prompt="describe", model="haiku", image_paths=(img1, img2)
    )
    text = _compose_stdin_prompt(r)
    assert str(img1) in text
    assert str(img2) in text
    assert "describe" in text
    assert "Read tool" in text


def test_all_dirs_dedupes_image_parents_and_extras(tmp_path: Path) -> None:
    d1 = tmp_path / "a"
    d1.mkdir()
    d2 = tmp_path / "b"
    d2.mkdir()
    img1 = d1 / "x.png"
    img1.write_bytes(_png_bytes((1, 1, 1)))
    img2 = d1 / "y.png"  # same dir as img1
    img2.write_bytes(_png_bytes((2, 2, 2)))
    img3 = d2 / "z.png"
    img3.write_bytes(_png_bytes((3, 3, 3)))

    dirs = _all_dirs([img1, img2, img3], [d2])  # d2 overlaps with img3 parent
    resolved = {str(p) for p in dirs}
    assert str(d1.resolve()) in resolved
    assert str(d2.resolve()) in resolved
    assert len(dirs) == 2


def test_file_sha_matches_known_content(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    # Known SHA-256 of "hello world"
    assert (
        _file_sha(p)
        == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_call_uses_cache_when_present(tmp_path: Path) -> None:
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"
    cached_text = "cached!"
    response_dir = cache_dir / "responses"
    response_dir.mkdir(parents=True)
    cache_path = response_dir / f"{_request_sha(r)}.json"
    cache_path.write_text(json.dumps({"text": cached_text}))

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess") as mock_sub:
            response = call(r)
    assert response.cached is True
    assert response.text == cached_text
    mock_sub.assert_not_called()


def test_call_writes_cache_after_fresh_run(tmp_path: Path) -> None:
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch(
            "_claude._run_claude_subprocess", return_value="fresh response"
        ):
            response = call(r)

    assert response.cached is False
    assert response.text == "fresh response"
    cache_path = cache_dir / "responses" / f"{_request_sha(r)}.json"
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text())
    assert payload["text"] == "fresh response"


def test_call_creates_cache_dir_if_missing(tmp_path: Path) -> None:
    r = ClaudeRequest(prompt="hi", model="haiku")
    cache_dir = tmp_path / "nested" / "cache"
    assert not cache_dir.exists()

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess", return_value="ok"):
            call(r)

    assert (cache_dir / "responses").exists()
    assert (cache_dir / "requests").exists()


def test_call_records_request_on_fresh_run(tmp_path: Path) -> None:
    """Every call writes the serialized request into requests/<sha>.json so
    a re-run is auditable from disk even before the response comes back.
    """
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess", return_value="ok"):
            call(r)

    request_path = cache_dir / "requests" / f"{_request_sha(r)}.json"
    assert request_path.exists()
    payload = json.loads(request_path.read_text())
    assert payload["prompt"] == "hi"
    assert payload["model"] == "haiku"
    # Path objects are serialized as strings via the json default=str hook.
    assert payload["image_paths"] == [str(img)]


def test_call_records_request_even_when_response_is_cached(
    tmp_path: Path,
) -> None:
    """A cache hit short-circuits the subprocess but still records the
    request, so the on-disk request log is complete regardless of whether
    each call hit the cache or the network.
    """
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"
    response_dir = cache_dir / "responses"
    response_dir.mkdir(parents=True)
    (response_dir / f"{_request_sha(r)}.json").write_text(
        json.dumps({"text": "cached!"})
    )

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess") as mock_sub:
            response = call(r)

    assert response.cached is True
    mock_sub.assert_not_called()
    request_path = cache_dir / "requests" / f"{_request_sha(r)}.json"
    assert request_path.exists()
    payload = json.loads(request_path.read_text())
    assert payload["prompt"] == "hi"


# ---------------------------------------------------------------------------
# Durable LLM records — cache → records → API lookup chain
# ---------------------------------------------------------------------------


def test_call_writes_durable_record_after_fresh_run(
    tmp_path: Path, isolate_llm_records: Path
) -> None:
    """A fresh API call produces both a cache entry and a durable record;
    the record is the source of truth that survives cache deletion.
    """
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch(
            "_claude._run_claude_subprocess", return_value="fresh response"
        ):
            response = call(r)

    assert response.text == "fresh response"
    record_path = isolate_llm_records / f"{_request_sha(r)}.json"
    assert record_path.exists()
    record = json.loads(record_path.read_text())
    assert record["request_sha"] == _request_sha(r)
    assert record["response"]["text"] == "fresh response"
    assert record["request"]["prompt"] == "hi"
    assert record["request"]["model"] == "haiku"
    # first_observed_at is an ISO-8601 timestamp; just assert it parses.
    datetime.fromisoformat(record["first_observed_at"])


def test_call_serves_from_records_when_cache_is_missing(
    tmp_path: Path, isolate_llm_records: Path
) -> None:
    """Delete the cache entirely; a durable record must satisfy the
    call without an API hit. This is the deletion-survival test —
    without it, the records layer isn't a real durable layer.
    """
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"
    # No cache directory at all — wholesale deletion.
    record_path = isolate_llm_records / f"{_request_sha(r)}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "request_sha": _request_sha(r),
                "request": {"prompt": "hi", "model": "haiku"},
                "response": {"text": "from records"},
                "first_observed_at": "2026-04-29T10:00:00-04:00",
            }
        )
    )

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess") as mock_sub:
            response = call(r)

    assert response.text == "from records"
    assert response.cached is True  # zero LLM cost paid
    mock_sub.assert_not_called()


def test_records_hit_repopulates_cache(
    tmp_path: Path, isolate_llm_records: Path
) -> None:
    """A records-only hit warms the cache so the next call short-circuits
    at the cache layer — same code path as a fresh API call would have
    produced. The cache is rebuildable from records, not a separate
    source of truth.
    """
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"
    record_path = isolate_llm_records / f"{_request_sha(r)}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "request_sha": _request_sha(r),
                "request": {"prompt": "hi", "model": "haiku"},
                "response": {"text": "from records"},
                "first_observed_at": "2026-04-29T10:00:00-04:00",
            }
        )
    )

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess") as mock_sub:
            call(r)
            # Second call: the cache should now be populated; subprocess
            # would have been called only on misses-everywhere.
            response2 = call(r)

    cache_path = cache_dir / "responses" / f"{_request_sha(r)}.json"
    assert cache_path.exists()
    assert json.loads(cache_path.read_text())["text"] == "from records"
    assert response2.text == "from records"
    mock_sub.assert_not_called()


def test_torn_cache_falls_through_to_records(
    tmp_path: Path, isolate_llm_records: Path
) -> None:
    """A partially-written cache file (e.g. process killed mid-write)
    must not bring down the next call. Treat it as a miss; records or
    the API supplies the actual answer.
    """
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"
    response_dir = cache_dir / "responses"
    response_dir.mkdir(parents=True)
    (response_dir / f"{_request_sha(r)}.json").write_text("{not valid json")
    record_path = isolate_llm_records / f"{_request_sha(r)}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "request_sha": _request_sha(r),
                "request": {"prompt": "hi", "model": "haiku"},
                "response": {"text": "from records"},
                "first_observed_at": "2026-04-29T10:00:00-04:00",
            }
        )
    )

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess") as mock_sub:
            response = call(r)
    assert response.text == "from records"
    mock_sub.assert_not_called()


def test_record_with_wrong_shape_treated_as_miss(
    tmp_path: Path, isolate_llm_records: Path
) -> None:
    """A records file at the right path but with the wrong JSON shape
    (e.g. a list at the top, or a dict missing the response object)
    is treated as a miss rather than crashing the call.
    """
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"
    record_path = isolate_llm_records / f"{_request_sha(r)}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    # Top-level array → not a dict; falls through.
    record_path.write_text(json.dumps([1, 2, 3]))

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess", return_value="api"):
            response = call(r)
    assert response.text == "api"

    # Now the right shape but missing the response.text — also a miss.
    record_path.write_text(json.dumps({"request_sha": "x", "request": {}}))
    # Wipe the cache that the previous run wrote so we re-enter the
    # records branch.
    cache_dir_inner = cache_dir / "responses"
    for p in cache_dir_inner.iterdir():
        p.unlink()
    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess", return_value="api2"):
            response2 = call(r)
    assert response2.text == "api2"


def test_torn_record_falls_through_to_api(
    tmp_path: Path, isolate_llm_records: Path
) -> None:
    """A torn record (similar crash scenario for the records write) is
    a miss too; the API call is the last-resort source of truth.
    """
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"
    record_path = isolate_llm_records / f"{_request_sha(r)}.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("{torn")

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess", return_value="api"):
            response = call(r)
    assert response.text == "api"
    # The torn record gets replaced by the new write — atomic temp+rename.
    new_record = json.loads(record_path.read_text())
    assert new_record["response"]["text"] == "api"


def test_record_written_before_cache(
    tmp_path: Path, isolate_llm_records: Path
) -> None:
    """The durable record must be on disk before the cache file. If the
    process crashes between the two writes we want the record present
    (truth) without a cache (a torn cache would just be a miss next
    time, recoverable from the record).
    """
    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(prompt="hi", model="haiku", image_paths=(img,))
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "responses" / f"{_request_sha(r)}.json"
    record_path = isolate_llm_records / f"{_request_sha(r)}.json"

    write_order: list[str] = []
    real_atomic_write = __import__("_claude")._atomic_write_json

    def tracking_write(path: Path, payload: object, **kwargs: object) -> None:
        if path == record_path:
            write_order.append("record")
        elif path == cache_path:
            write_order.append("cache")
        real_atomic_write(path, payload, **kwargs)  # type: ignore[arg-type]

    with patch("_claude.CACHE_DIR", cache_dir):
        with patch("_claude._run_claude_subprocess", return_value="api"):
            with patch(
                "_claude._atomic_write_json", side_effect=tracking_write
            ):
                call(r)
    # Record must come first.
    assert write_order == ["record", "cache"]


def test_atomic_write_leaves_no_tmp_on_success(
    tmp_path: Path,
) -> None:
    """``_atomic_write_json`` must rename the temp file in place; no
    leftover ``.tmp`` files in the destination directory.
    """
    from _claude import _atomic_write_json

    target = tmp_path / "out.json"
    _atomic_write_json(target, {"x": 1})
    assert target.exists()
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_run_claude_subprocess_raises_on_nonzero_exit() -> None:
    from _claude import _run_claude_subprocess

    r = ClaudeRequest(prompt="hi", model="haiku")

    class FakeCompleted:
        returncode = 1
        stdout = ""
        stderr = "boom"

    with patch("_claude.subprocess.run", return_value=FakeCompleted()):
        with pytest.raises(ClaudeSubprocessError) as excinfo:
            _run_claude_subprocess(r)
    assert excinfo.value.returncode == 1
    assert "boom" in str(excinfo.value)


def test_detect_rate_limit_json_envelope() -> None:
    envelope = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 429,
            "result": "You've hit your limit · resets 1am (America/New_York)",
        }
    )
    msg = _detect_rate_limit(envelope, "")
    assert msg is not None
    assert "1am" in msg


def test_detect_rate_limit_text_mode_substring() -> None:
    msg = _detect_rate_limit("You've hit your limit · resets soon\n", "")
    assert msg is not None
    assert "hit your limit" in msg.lower()


def test_detect_rate_limit_text_mode_substring_only_in_stderr() -> None:
    msg = _detect_rate_limit("", "Error: you've hit your limit\n")
    assert msg is not None


def test_detect_rate_limit_text_mode_no_per_line_match() -> None:
    """Defensive: even if line-splitting fails to find the marker, the
    substring path returns the generic message rather than None.
    """
    # Force a case where the marker is in stdout but split() yields the
    # only line — already covered by the substring check at end.
    msg = _detect_rate_limit("hit your limit", "")
    assert msg is not None


def test_detect_rate_limit_returns_none_for_normal_response() -> None:
    envelope = json.dumps(
        {"type": "result", "is_error": False, "result": "all good"}
    )
    assert _detect_rate_limit(envelope, "") is None


def test_detect_rate_limit_returns_none_for_non_json_normal_text() -> None:
    assert _detect_rate_limit("hello world", "") is None


def test_detect_rate_limit_envelope_other_error_is_not_rate_limit() -> None:
    """A 500 error envelope shouldn't be classified as rate limit."""
    envelope = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 500,
            "result": "internal server error",
        }
    )
    assert _detect_rate_limit(envelope, "") is None


def test_run_claude_subprocess_raises_rate_limit_on_429_envelope() -> None:
    from _claude import _run_claude_subprocess

    r = ClaudeRequest(
        prompt="hi",
        model="sonnet",
        json_schema='{"type":"object"}',
    )

    envelope = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 429,
            "result": "You've hit your limit · resets 1am",
        }
    )

    class FakeCompleted:
        returncode = 1
        stdout = envelope
        stderr = ""

    with patch("_claude.subprocess.run", return_value=FakeCompleted()):
        with pytest.raises(ClaudeRateLimitError) as excinfo:
            _run_claude_subprocess(r)
    assert "1am" in excinfo.value.limit_message
    # Subclass relationship — callers that catch ClaudeSubprocessError still see it.
    assert isinstance(excinfo.value, ClaudeSubprocessError)
    # __str__ surfaces the parsed message verbatim.
    assert "1am" in str(excinfo.value)


# ---------- rate-limit parser + retry loop -------------------------------


def test_parse_reset_time_basic_am() -> None:
    now = datetime(2026, 4, 26, 23, 0, tzinfo=ZoneInfo("America/New_York"))
    target = _parse_reset_time(
        "You've hit your limit · resets 1am (America/New_York)", now=now
    )
    assert target is not None
    assert target.hour == 1 and target.minute == 0
    assert target.tzinfo is not None
    # 23:00 today → next 1am is tomorrow's 01:00.
    delta = (target - now).total_seconds()
    assert 1.5 * 3600 < delta < 2.5 * 3600


def test_parse_reset_time_pm_with_timezone() -> None:
    now = datetime(2026, 4, 26, 9, 0, tzinfo=ZoneInfo("Europe/Berlin"))
    target = _parse_reset_time("resets 4pm (Europe/Berlin)", now=now)
    assert target is not None
    assert target.hour == 16


def test_parse_reset_time_handles_12am_and_12pm() -> None:
    now = datetime(2026, 4, 26, 9, 0, tzinfo=ZoneInfo("UTC"))
    midnight = _parse_reset_time("resets 12am (UTC)", now=now)
    noon = _parse_reset_time("resets 12pm (UTC)", now=now)
    assert midnight is not None and midnight.hour == 0
    assert noon is not None and noon.hour == 12


def test_parse_reset_time_with_minutes() -> None:
    now = datetime(2026, 4, 26, 9, 0, tzinfo=ZoneInfo("UTC"))
    target = _parse_reset_time("resets 1:30pm (UTC)", now=now)
    assert target is not None
    assert target.hour == 13 and target.minute == 30


def test_parse_reset_time_picks_nearest_occurrence_for_recently_passed_reset() -> (
    None
):
    """Server lag scenario: now=01:05am, message says 'resets 1am'. Picking
    'today's 01:00' (just past) is nearer than 'tomorrow's 01:00', and the
    sleep computation will then floor to 60s.
    """
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 4, 27, 1, 5, tzinfo=tz)
    target = _parse_reset_time("resets 1am (America/New_York)", now=now)
    assert target is not None
    # Today's 01:00 (5 minutes before now), not tomorrow's.
    delta = (target - now).total_seconds()
    assert delta < 0
    assert abs(delta) < 600  # within ~10 minutes of now


def test_parse_reset_time_with_date_prefix() -> None:
    tz = ZoneInfo("Africa/Libreville")
    now = datetime(2026, 2, 19, 9, 0, tzinfo=tz)
    target = _parse_reset_time(
        "resets Feb 20, 5pm (Africa/Libreville)", now=now
    )
    assert target is not None
    assert target.month == 2 and target.day == 20 and target.hour == 17


def test_parse_reset_time_dated_in_future_keeps_current_year() -> None:
    """A 'Dec 31' message read in February uses the current year, not next."""
    tz = ZoneInfo("UTC")
    now = datetime(2026, 2, 1, 9, 0, tzinfo=tz)
    target = _parse_reset_time("resets Dec 31, 5pm (UTC)", now=now)
    assert target is not None
    assert target.year == 2026 and target.month == 12 and target.day == 31


def test_parse_reset_time_dated_far_in_past_rolls_to_next_year() -> None:
    """A 'Jan 1' message read in late December has target = THIS year's Jan 1
    on the first parse, which is ~11 months in the past — advance to next
    year.
    """
    tz = ZoneInfo("UTC")
    now = datetime(2026, 12, 30, 9, 0, tzinfo=tz)
    target = _parse_reset_time("resets Jan 1, 1pm (UTC)", now=now)
    assert target is not None
    assert target.year == 2027 and target.month == 1 and target.day == 1


def test_parse_reset_time_returns_none_for_unmatched_message() -> None:
    assert _parse_reset_time("some unrelated error") is None


def test_parse_reset_time_returns_none_for_invalid_timezone() -> None:
    assert _parse_reset_time("resets 1am (Not/AnyTimezone)") is None


def test_parse_reset_time_returns_none_for_invalid_dated_calendar() -> None:
    """Feb 30 doesn't exist; parser should bail rather than fabricate."""
    now = datetime(2026, 2, 1, 9, 0, tzinfo=ZoneInfo("UTC"))
    assert _parse_reset_time("resets Feb 30, 5pm (UTC)", now=now) is None


def test_compute_sleep_seconds_clamps_to_minimum() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 4, 27, 1, 5, tzinfo=tz)
    sleep = _compute_rate_limit_sleep_seconds(
        "resets 1am (America/New_York)", now=now
    )
    assert sleep == MIN_RATE_LIMIT_SLEEP_SECONDS


def test_compute_sleep_seconds_uses_target_when_in_future() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 4, 26, 23, 0, tzinfo=tz)
    sleep = _compute_rate_limit_sleep_seconds(
        "resets 1am (America/New_York)", now=now
    )
    # 23:00 → next 01:00 is 2 hours away.
    assert 1.9 * 3600 < sleep < 2.1 * 3600


def test_compute_sleep_seconds_falls_back_when_unparseable() -> None:
    sleep = _compute_rate_limit_sleep_seconds("garbled message no reset info")
    assert sleep == RATE_LIMIT_FALLBACK_SLEEP_SECONDS


def test_call_retries_after_rate_limit_and_succeeds(tmp_path: Path) -> None:
    """First subprocess call hits 429; wrapper sleeps and retries; second
    call succeeds. time.sleep is mocked so the test stays fast.
    """
    cache_dir = tmp_path / "cache"
    rate_limit_envelope = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 429,
            "result": "You've hit your limit · resets 1am (America/New_York)",
        }
    )

    class FakeBusy:
        returncode = 1
        stdout = rate_limit_envelope
        stderr = ""

    class FakeOk:
        returncode = 0
        stdout = json.dumps(
            {"type": "result", "structured_output": {"answer": 42}}
        )
        stderr = ""

    sleeps: list[float] = []

    with (
        patch("_claude.CACHE_DIR", cache_dir),
        patch("_claude.subprocess.run", side_effect=[FakeBusy(), FakeOk()]),
        patch("_claude.time.sleep", side_effect=sleeps.append),
    ):
        response = call(
            ClaudeRequest(
                prompt="hi",
                model="sonnet",
                json_schema='{"type": "object"}',
            )
        )
    assert json.loads(response.text) == {"answer": 42}
    assert response.cached is False
    assert len(sleeps) == 1
    # We slept for a positive interval that respects the 60-second floor.
    assert sleeps[0] >= MIN_RATE_LIMIT_SLEEP_SECONDS


def test_call_gives_up_after_max_retries(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    rate_limit_envelope = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "api_error_status": 429,
            "result": "You've hit your limit · resets 1am (America/New_York)",
        }
    )

    class FakeBusy:
        returncode = 1
        stdout = rate_limit_envelope
        stderr = ""

    with (
        patch("_claude.CACHE_DIR", cache_dir),
        patch(
            "_claude.subprocess.run",
            side_effect=[FakeBusy()] * (MAX_RATE_LIMIT_RETRIES + 1),
        ),
        patch("_claude.time.sleep"),
    ):
        with pytest.raises(ClaudeRateLimitError):
            call(
                ClaudeRequest(
                    prompt="hi",
                    model="sonnet",
                    json_schema='{"type": "object"}',
                )
            )


def test_detect_structured_output_failure_with_message() -> None:
    envelope = json.dumps(
        {
            "type": "result",
            "subtype": "error_max_structured_output_retries",
            "is_error": True,
            "errors": [
                "Failed to provide valid structured output after 5 attempts"
            ],
        }
    )
    msg = _detect_structured_output_failure(envelope)
    assert msg is not None and "5 attempts" in msg


def test_detect_structured_output_failure_falls_back_to_default_message() -> (
    None
):
    """If the envelope has the right subtype but no errors list, surface a
    sensible default rather than None.
    """
    envelope = json.dumps(
        {
            "type": "result",
            "subtype": "error_max_structured_output_retries",
            "is_error": True,
        }
    )
    msg = _detect_structured_output_failure(envelope)
    assert msg is not None
    assert "structured output" in msg.lower()


def test_detect_structured_output_failure_returns_none_for_normal_response() -> (
    None
):
    envelope = json.dumps({"type": "result", "is_error": False})
    assert _detect_structured_output_failure(envelope) is None


def test_detect_structured_output_failure_returns_none_for_other_error_subtype() -> (
    None
):
    envelope = json.dumps(
        {"type": "result", "is_error": True, "subtype": "something_else"}
    )
    assert _detect_structured_output_failure(envelope) is None


def test_detect_structured_output_failure_returns_none_for_non_json() -> None:
    assert _detect_structured_output_failure("not json") is None


def test_detect_structured_output_failure_returns_none_for_non_object_envelope() -> (
    None
):
    assert _detect_structured_output_failure("[1, 2, 3]") is None


def test_run_claude_subprocess_raises_structured_output_error() -> None:
    from _claude import _run_claude_subprocess

    r = ClaudeRequest(prompt="x", model="opus", json_schema='{"type":"object"}')
    envelope = json.dumps(
        {
            "type": "result",
            "subtype": "error_max_structured_output_retries",
            "is_error": True,
            "errors": [
                "Failed to provide valid structured output after 5 attempts"
            ],
        }
    )

    class FakeCompleted:
        returncode = 1
        stdout = envelope
        stderr = ""

    with patch("_claude.subprocess.run", return_value=FakeCompleted()):
        with pytest.raises(ClaudeStructuredOutputError) as excinfo:
            _run_claude_subprocess(r)
    assert "5 attempts" in excinfo.value.failure_message
    # Subclass relationship — generic catchers still see it.
    assert isinstance(excinfo.value, ClaudeSubprocessError)
    assert "structured-output" in str(excinfo.value)


def test_run_claude_subprocess_returns_stripped_text_without_schema() -> None:
    """No json_schema → wrapper passes through model's text output (stripped)."""
    from _claude import _run_claude_subprocess

    r = ClaudeRequest(prompt="hi", model="haiku")

    class FakeCompleted:
        returncode = 0
        stdout = "  hello\n"
        stderr = ""

    with patch("_claude.subprocess.run", return_value=FakeCompleted()):
        result = _run_claude_subprocess(r)
    assert result == "hello"


def test_run_claude_subprocess_extracts_structured_output_when_schema_set(
    tmp_path: Path,
) -> None:
    """With json_schema set, the wrapper switches to --output-format json and
    pulls `structured_output` out of the envelope.
    """
    from _claude import _run_claude_subprocess

    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(
        prompt="x",
        model="sonnet",
        image_paths=(img,),
        json_schema='{"type":"object"}',
    )

    envelope = json.dumps(
        {
            "type": "result",
            "result": "prose",
            "structured_output": {"answer": 42},
        }
    )

    class FakeCompleted:
        returncode = 0
        stdout = envelope
        stderr = ""

    with patch("_claude.subprocess.run", return_value=FakeCompleted()):
        result_text = _run_claude_subprocess(r)
    assert json.loads(result_text) == {"answer": 42}


def test_run_claude_subprocess_raises_when_envelope_missing_structured_output(
    tmp_path: Path,
) -> None:
    from _claude import _run_claude_subprocess

    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(
        prompt="x",
        model="sonnet",
        image_paths=(img,),
        json_schema='{"type":"object"}',
    )

    class FakeCompleted:
        returncode = 0
        stdout = json.dumps({"type": "result", "result": "no schema field"})
        stderr = ""

    with patch("_claude.subprocess.run", return_value=FakeCompleted()):
        with pytest.raises(ClaudeSubprocessError, match="structured_output"):
            _run_claude_subprocess(r)


def test_run_claude_subprocess_builds_expected_argv(tmp_path: Path) -> None:
    from _claude import _run_claude_subprocess

    img = _make_image(tmp_path, "a.png", (255, 0, 0))
    r = ClaudeRequest(
        prompt="hi",
        model="sonnet",
        image_paths=(img,),
        system_prompt="be terse",
        json_schema='{"type":"string"}',
        extra_dirs=(tmp_path,),
    )

    captured: dict[str, object] = {}

    class FakeCompleted:
        returncode = 0
        stdout = json.dumps(
            {"type": "result", "structured_output": "out", "result": "x"}
        )
        stderr = ""

    def fake_run(cmd: list[str], **kwargs: object) -> FakeCompleted:
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return FakeCompleted()

    with patch("_claude.subprocess.run", side_effect=fake_run):
        result = _run_claude_subprocess(r)

    assert json.loads(result) == "out"
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "claude"
    assert "--model" in cmd and "sonnet" in cmd
    assert "--append-system-prompt" in cmd and "be terse" in cmd
    assert "--json-schema" in cmd
    # When json_schema is set we pin output-format to json so the envelope
    # contains structured_output.
    assert "--output-format" in cmd and "json" in cmd
    assert "--add-dir" in cmd
    assert isinstance(captured["input"], str) and "hi" in captured["input"]


def test_claude_response_dataclass_fields() -> None:
    r = ClaudeResponse(
        text="ok", cached=False, elapsed_seconds=1.5, request_sha="abc"
    )
    assert r.text == "ok"
    assert r.cached is False
    assert r.elapsed_seconds == 1.5
    assert r.request_sha == "abc"


def test_claude_subprocess_error_str_contains_components() -> None:
    err = ClaudeSubprocessError(
        returncode=2,
        stdout="stdout-line",
        stderr="stderr-line",
        cmd=["claude", "-p"],
    )
    s = str(err)
    assert "exited 2" in s
    assert "stdout-line" in s
    assert "stderr-line" in s
    assert "claude -p" in s


def test_main_cli_runs_call_and_prints(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _claude import main

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr("_claude.CACHE_DIR", cache_dir)
    monkeypatch.setattr("_claude._run_claude_subprocess", lambda req: "MAIN OK")
    monkeypatch.setattr("sys.argv", ["_claude.py", "haiku", "say hi"])
    monkeypatch.setenv("CLAUDE_QUIET", "1")
    rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "MAIN OK" in out


def test_main_cli_prints_request_sha_when_not_quiet(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without CLAUDE_QUIET the wrapper prints a [cached] / [Ns] prefix line."""
    from _claude import main

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr("_claude.CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        "_claude._run_claude_subprocess", lambda req: "second response"
    )
    monkeypatch.setattr("sys.argv", ["_claude.py", "haiku", "different"])
    monkeypatch.delenv("CLAUDE_QUIET", raising=False)
    rc = main()
    assert rc == 0
    out = capsys.readouterr().out
    # Expect two lines: the prefix line and the model output.
    lines = out.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("[")  # either [cached] or [Ns]
    assert lines[1] == "second response"
