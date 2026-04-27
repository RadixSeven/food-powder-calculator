"""Subprocess wrapper around `claude -p` with caching.

All pipeline stages route their LLM calls through this module so the
subprocess invocation, retry, and cache live in exactly one place. Tests
mock `_run_claude_subprocess` to avoid network access.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"
# Subdir of CACHE_DIR that records every request we send (independent of
# whether a response is cached), so a re-run can be audited from disk.
# Derived from CACHE_DIR at call time so tests that patch CACHE_DIR pick
# up the override automatically.
REQUEST_CACHE_SUBDIR = "requests"

# Wall-clock cap per claude invocation. Long enough for vision calls on
# large images, short enough that a hung subprocess doesn't block forever.
DEFAULT_TIMEOUT_SECONDS = 180

# Minimum sleep between rate-limit retries so we don't pound the server even
# if the parsed reset time has already passed (server-side lag is a known
# issue — see anthropics/claude-code#20719).
MIN_RATE_LIMIT_SLEEP_SECONDS = 60.0
# Fallback if the limit message can't be parsed.
RATE_LIMIT_FALLBACK_SLEEP_SECONDS = 3600.0
# Cap how many rate-limit retries we'll do in one call before giving up; this
# bounds blast radius if parsing or the server are misbehaving.
MAX_RATE_LIMIT_RETRIES = 6


@dataclass(frozen=True)
class ClaudeRequest:
    """One LLM call: a prompt, optional images, model, and any extra dirs."""

    prompt: str
    model: str  # "haiku", "sonnet", "opus", or a full model ID
    image_paths: tuple[Path, ...] = ()
    json_schema: str | None = None
    system_prompt: str | None = None
    extra_dirs: tuple[Path, ...] = ()
    # Bumped when we change wrapper behavior in a way that should
    # invalidate cached responses (parser changes, schema reorganization).
    cache_version: int = 1


@dataclass
class ClaudeResponse:
    """The text the model returned, plus bookkeeping."""

    text: str
    cached: bool
    elapsed_seconds: float = 0.0
    request_sha: str = ""


def call(request: ClaudeRequest) -> ClaudeResponse:
    """Run a single claude -p call, returning a cached response if available.

    Rate-limit responses (HTTP 429) trigger an in-process wait until the
    parsed reset time and a retry, so the caller doesn't have to know about
    rate limits — particularly important when the caller is itself an LLM
    agent. Each retry sleeps at least :data:`MIN_RATE_LIMIT_SLEEP_SECONDS`
    so a stale-clock or lagging-server condition can't make the loop hot.

    """
    request_sha = _request_sha(request)
    request_cache_dir = CACHE_DIR / REQUEST_CACHE_SUBDIR
    request_cache_dir.mkdir(parents=True, exist_ok=True)
    request_cache_path = request_cache_dir / f"{request_sha}.json"
    cache_path = CACHE_DIR / f"{request_sha}.json"
    with request_cache_path.open("w") as req_cache_file:
        json.dump(asdict(request), req_cache_file, default=str)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        return ClaudeResponse(
            text=payload["text"],
            cached=True,
            elapsed_seconds=0.0,
            request_sha=request_sha,
        )

    start = time.monotonic()
    text = _run_with_rate_limit_retry(request)
    elapsed = time.monotonic() - start

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"text": text, "request_sha": request_sha}, indent=2)
    )
    return ClaudeResponse(
        text=text,
        cached=False,
        elapsed_seconds=elapsed,
        request_sha=request_sha,
    )


def _run_with_rate_limit_retry(request: ClaudeRequest) -> str:
    """Invoke the subprocess with auto-wait-and-retry on rate-limit errors."""
    last_error: ClaudeRateLimitError | None = None
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return _run_claude_subprocess(request)
        except ClaudeRateLimitError as e:
            last_error = e
            sleep_seconds = _compute_rate_limit_sleep_seconds(e.limit_message)
            now = datetime.now().astimezone()
            wake_at = now + timedelta(seconds=sleep_seconds)
            print(
                f"[claude rate-limit] {e.limit_message}\n"
                f"[claude rate-limit] sleeping {sleep_seconds:.0f}s until "
                f"{wake_at.isoformat(timespec='seconds')} "
                f"(attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_seconds)
            print(
                "[claude rate-limit] retrying after sleep",
                file=sys.stderr,
                flush=True,
            )
    assert last_error is not None
    raise last_error


_RATE_LIMIT_TIME_RE = re.compile(
    r"resets\s+"
    r"(?:(\w+)\s+(\d{1,2}),\s+)?"  # optional "Feb 20, "
    r"(\d{1,2})(?::(\d{2}))?\s*"  # hour with optional ":MM"
    r"(am|pm)\s*"  # am or pm
    r"\(([^)]+)\)",  # timezone in parens
    re.IGNORECASE,
)


def _parse_reset_time(
    message: str, *, now: datetime | None = None
) -> datetime | None:
    """Parse the reset target from a Claude Code rate-limit message.

    Handles every format observed in the wild as of 2026-04:

    * ``resets 1am (America/New_York)``
    * ``resets 4pm (Europe/Berlin)``
    * ``resets 2pm (UTC)``
    * ``resets 11pm (America/Anchorage)``
    * ``resets Feb 20, 5pm (Africa/Libreville)``

    For the hour-only forms we pick the nearest occurrence (yesterday,
    today, or tomorrow at the parsed clock time in the named tz). That way
    server lag past a recent reset doesn't cause us to wait 24 hours for
    "tomorrow's" 1am — :func:`_compute_rate_limit_sleep_seconds` clamps to
    a 60-second floor regardless.

    Returns ``None`` if the message doesn't match any known shape.
    """
    m = _RATE_LIMIT_TIME_RE.search(message)
    if m is None:
        return None
    month_abbrev, day_s, hour_s, min_s, ampm, tz_name = m.groups()
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, KeyError):
        return None
    hour = int(hour_s) % 12
    if ampm.lower() == "pm":
        hour += 12
    minute = int(min_s) if min_s else 0
    if now is None:
        now = datetime.now(tz=tz)
    else:
        now = now.astimezone(tz)
    if month_abbrev:
        try:
            target_date = datetime.strptime(
                f"{month_abbrev} {day_s} {now.year}", "%b %d %Y"
            ).date()
        except ValueError:
            return None
        target = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            hour,
            minute,
            tzinfo=tz,
        )
        # If the dated form fell more than 6 months in the past it's almost
        # certainly meant for next year (e.g., a December message read in
        # January).
        if target < now - timedelta(days=183):
            target = target.replace(year=target.year + 1)
        return target
    today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    candidates = [today - timedelta(days=1), today, today + timedelta(days=1)]
    return min(candidates, key=lambda c: abs((c - now).total_seconds()))


def _compute_rate_limit_sleep_seconds(
    message: str, *, now: datetime | None = None
) -> float:
    """Seconds to sleep before retrying the rate-limited call.

    Always at least :data:`MIN_RATE_LIMIT_SLEEP_SECONDS`. If the message
    can't be parsed at all, falls back to
    :data:`RATE_LIMIT_FALLBACK_SLEEP_SECONDS`.
    """
    target = _parse_reset_time(message, now=now)
    if target is None:
        return RATE_LIMIT_FALLBACK_SLEEP_SECONDS
    if now is None:
        now = datetime.now(tz=target.tzinfo)
    delta = (target - now).total_seconds()
    return max(MIN_RATE_LIMIT_SLEEP_SECONDS, delta)


def _run_claude_subprocess(request: ClaudeRequest) -> str:
    """Invoke `claude -p` with the request's settings.

    Images and prompt go through stdin; model + add-dir flags are CLI args.
    When ``json_schema`` is set we switch the wire format to JSON envelope so
    we can pull the schema-validated ``structured_output`` field out — in
    plain-text mode the structured output is dropped.

    Rate-limit failures (HTTP 429) are detected in the JSON envelope (or in
    stdout text in text-mode) and raised as :class:`ClaudeRateLimitError`,
    which is a subclass of :class:`ClaudeSubprocessError` so callers that
    only catch the latter still see them. The wrapper does NOT retry — that
    would burn more LLM calls — leaving cleanup (e.g. exit-and-resume from
    the cache) to the caller.
    """
    cmd: list[str] = [
        "claude",
        "--no-session-persistence",
        "-p",
        "--model",
        request.model,
    ]
    for d in _all_dirs(request.image_paths, request.extra_dirs):
        cmd.extend(["--add-dir", str(d)])
    if request.system_prompt is not None:
        cmd.extend(["--append-system-prompt", request.system_prompt])
    if request.json_schema is not None:
        cmd.extend(["--json-schema", request.json_schema])
        cmd.extend(["--output-format", "json"])

    prompt_text = _compose_stdin_prompt(request)
    result = subprocess.run(  # noqa: S603 — args constructed from typed fields
        cmd,
        input=prompt_text,
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        check=False,
    )
    rate_limit_message = _detect_rate_limit(result.stdout, result.stderr)
    if rate_limit_message is not None:
        raise ClaudeRateLimitError(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            cmd=cmd,
            limit_message=rate_limit_message,
        )
    structured_failure = _detect_structured_output_failure(result.stdout)
    if structured_failure is not None:
        raise ClaudeStructuredOutputError(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            cmd=cmd,
            failure_message=structured_failure,
        )
    if result.returncode != 0:
        raise ClaudeSubprocessError(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            cmd=cmd,
        )
    if request.json_schema is not None:
        return _extract_structured_output(result.stdout)
    return result.stdout.strip()


def _detect_structured_output_failure(stdout: str) -> str | None:
    """Return the failure message if claude exhausted its structured-output
    retries (subtype ``error_max_structured_output_retries``), else None.

    This is a model-side failure mode distinct from rate limiting: when
    asked for ``--json-schema`` output, claude internally retries when its
    response doesn't validate, and gives up after 5 attempts. Common at
    small image sizes where the model can't read enough to fill the schema.
    """
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(envelope, dict):
        return None
    if not envelope.get("is_error"):
        return None
    subtype = envelope.get("subtype", "")
    if not (
        isinstance(subtype, str)
        and subtype.startswith("error_max_structured_output")
    ):
        return None
    errors = envelope.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], str):
        return errors[0]
    return "structured output retries exhausted"


def _detect_rate_limit(stdout: str, stderr: str) -> str | None:
    """Return the rate-limit message if claude reported 429, else None.

    Two transports to handle:

    * JSON envelope mode (``--output-format json``): ``is_error: true`` AND
      ``api_error_status: 429`` in the parsed envelope.
    * Plain-text mode: stdout/stderr contains "hit your limit" or a literal
      "429" near "limit" — Claude Code's rate-limit message is human-
      readable and stable enough for substring matching.
    """
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        envelope = None
    if isinstance(envelope, dict):
        if envelope.get("is_error") and envelope.get("api_error_status") == 429:
            return str(envelope.get("result") or "rate limit hit")
    for line in (stdout + "\n" + stderr).splitlines():
        if "hit your limit" in line.lower():
            return line.strip()
    return None


def _extract_structured_output(envelope_json: str) -> str:
    """Pull `structured_output` from the JSON envelope as a JSON-serialized string.

    With ``--output-format json``, ``claude -p`` prints a single JSON object
    whose ``structured_output`` field holds the schema-validated model
    response. Callers expect a JSON string so we re-serialize that field
    back into text.
    """
    envelope = json.loads(envelope_json)
    if "structured_output" not in envelope:
        raise ClaudeSubprocessError(
            returncode=0,
            stdout=envelope_json,
            stderr="claude returned JSON envelope without structured_output",
            cmd=[],
        )
    return json.dumps(envelope["structured_output"])


def _compose_stdin_prompt(request: ClaudeRequest) -> str:
    """Build the stdin payload: image refs first, then the prompt."""
    if not request.image_paths:
        return request.prompt
    image_block = "\n".join(
        f"Image {i + 1}: {p}" for i, p in enumerate(request.image_paths)
    )
    return (
        "The following images are at the listed paths. Use the Read tool to "
        "view each one before answering.\n\n"
        f"{image_block}\n\n"
        f"{request.prompt}"
    )


def _all_dirs(
    image_paths: Iterable[Path], extra_dirs: Iterable[Path]
) -> list[Path]:
    """Distinct parent dirs of the images, plus any explicitly-allowed dirs."""
    seen: dict[str, Path] = {}
    for p in image_paths:
        d = p.resolve().parent
        seen[str(d)] = d
    for d in extra_dirs:
        rd = d.resolve()
        seen[str(rd)] = rd
    return sorted(seen.values(), key=str)


def _request_sha(request: ClaudeRequest) -> str:
    """Stable hash of everything that could change the response."""
    h = hashlib.sha256()
    h.update(request.model.encode())
    h.update(b"\0")
    h.update(request.prompt.encode())
    h.update(b"\0")
    h.update((request.system_prompt or "").encode())
    h.update(b"\0")
    h.update((request.json_schema or "").encode())
    h.update(b"\0")
    h.update(str(request.cache_version).encode())
    h.update(b"\0")
    for p in request.image_paths:
        h.update(_file_sha(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:32]


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ClaudeSubprocessError(Exception):
    """Raised when the `claude` subprocess returns non-zero."""

    returncode: int
    stdout: str
    stderr: str
    cmd: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"claude exited {self.returncode}\n"
            f"cmd: {' '.join(self.cmd)}\n"
            f"stderr: {self.stderr.strip()}\n"
            f"stdout: {self.stdout.strip()}"
        )


@dataclass
class ClaudeRateLimitError(ClaudeSubprocessError):
    """Raised by ``_run_claude_subprocess`` on an HTTP 429 response.

    The high-level :func:`call` catches these and sleeps until the parsed
    reset time before retrying, so most callers will never see this
    exception. It only escapes :func:`call` if the retry budget
    (:data:`MAX_RATE_LIMIT_RETRIES`) is exhausted.
    """

    limit_message: str = ""

    def __str__(self) -> str:
        return f"claude rate-limit: {self.limit_message}"


@dataclass
class ClaudeStructuredOutputError(ClaudeSubprocessError):
    """Raised when claude exhausted its internal structured-output retries.

    Indicates the model couldn't produce schema-valid JSON for the call —
    typically at very low image resolutions where there isn't enough text
    to fill the schema. Callers that drive a search across multiple
    parameter values (e.g. binary-search sizing probes) can treat this as
    a clean "no" for that probe rather than as a fatal subprocess error.
    """

    failure_message: str = ""

    def __str__(self) -> str:
        return f"claude structured-output retries exhausted: {self.failure_message}"


# Re-exports
__all__ = [
    "CACHE_DIR",
    "ClaudeRateLimitError",
    "ClaudeRequest",
    "ClaudeResponse",
    "ClaudeStructuredOutputError",
    "ClaudeSubprocessError",
    "call",
]


def main() -> int:
    """Tiny CLI: `python scripts/_claude.py <model> <prompt>` for ad-hoc use."""
    import argparse

    parser = argparse.ArgumentParser(
        description="One-shot claude -p call via the cached wrapper."
    )
    parser.add_argument("model", help="haiku, sonnet, opus, or full model ID")
    parser.add_argument("prompt", help="Prompt text")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Path to image (repeatable)",
    )
    args = parser.parse_args()

    response = call(
        ClaudeRequest(
            prompt=args.prompt,
            model=args.model,
            image_paths=tuple(Path(p) for p in args.image),
        )
    )
    if os.environ.get("CLAUDE_QUIET", "").lower() not in {"1", "true", "yes"}:
        prefix = (
            "[cached]"
            if response.cached
            else f"[{response.elapsed_seconds:.1f}s]"
        )
        print(prefix, response.request_sha)
    print(response.text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
