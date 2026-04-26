"""Subprocess wrapper around `claude -p` with caching.

All pipeline stages route their LLM calls through this module so the
subprocess invocation, retry, and cache live in exactly one place. Tests
mock `_run_claude_subprocess` to avoid network access.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"

# Wall-clock cap per claude invocation. Long enough for vision calls on
# large images, short enough that a hung subprocess doesn't block forever.
DEFAULT_TIMEOUT_SECONDS = 180


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
    """Run a single claude -p call, returning a cached response if available."""
    request_sha = _request_sha(request)
    cache_path = CACHE_DIR / f"{request_sha}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        return ClaudeResponse(
            text=payload["text"],
            cached=True,
            elapsed_seconds=0.0,
            request_sha=request_sha,
        )

    start = time.monotonic()
    text = _run_claude_subprocess(request)
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


def _run_claude_subprocess(request: ClaudeRequest) -> str:
    """Invoke `claude -p` with the request's settings.

    Images and prompt go through stdin; model + add-dir flags are CLI args.
    When ``json_schema`` is set we switch the wire format to JSON envelope so
    we can pull the schema-validated ``structured_output`` field out — in
    plain-text mode the structured output is dropped.
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


# Re-exports
__all__ = [
    "CACHE_DIR",
    "ClaudeRequest",
    "ClaudeResponse",
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
