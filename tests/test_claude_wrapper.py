"""Tests for scripts/_claude.py — the cached `claude -p` subprocess wrapper.

The actual `claude` subprocess is mocked everywhere; these tests run with
no network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from _claude import (
    ClaudeRequest,
    ClaudeResponse,
    ClaudeSubprocessError,
    _all_dirs,
    _compose_stdin_prompt,
    _file_sha,
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
    cache_path = cache_dir / f"{_request_sha(r)}.json"
    cache_dir.mkdir()
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
    cache_path = cache_dir / f"{_request_sha(r)}.json"
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

    assert cache_dir.exists()


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
    pulls `structured_output` out of the envelope."""
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
