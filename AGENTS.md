# Agent instructions

## Tests and static analysis

Run all four before committing — they must all pass, tests must have 100%
coverage, and the tree must be fully formatted:

```shell
pants test ::              # tests + coverage (HiGHS/pyomo deps declared in tests/BUILD)
uvx ruff format .          # format
uvx ruff check --fix .     # lint and auto-fix
uvx pyrefly check          # type-check
```

## Environment quirks

- The project's `.venv/` is intentionally minimal — runtime deps live in
  pants' own pex venvs under `~/.cache/pants/named_caches/pex_root/`.
- `pyrefly` resolves imports through `.venv/`, so it needs `pyomo`,
  `highspy`, and `pytest` installed there. If pyrefly reports
  `missing-import`, install them with:

  ```shell
  uv pip install --python .venv/bin/python pyomo highspy pytest
  ```

- Pyrefly config lives in `pyproject.toml` under `[tool.pyrefly]` and adds
  `stubs/` to the search path so the local `pyomo.environ` Protocol stubs in
  `stubs/pyomo/environ.pyi` are picked up.
- The default pants cache (`~/.cache/pants`) works fine; no need to redirect
  it into the working directory.

## Commit checklist

Before `git commit`:

1. `pants test ::` is green and shows **100%** coverage across `src/` and
   `tests/`.
2. `uvx ruff format .` reports `N files left unchanged` (no further
   reformatting needed).
3. `uvx ruff check --fix .` reports `All checks passed!`.
4. `uvx pyrefly check` reports `0 errors`.
