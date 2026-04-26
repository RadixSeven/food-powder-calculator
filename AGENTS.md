# Agent instructions

## Tests and static analysis

Run all four before committing — they must all pass, tests must have 100%
coverage, and the tree must be fully formatted:

```shell
pants test ::                 # tests + coverage (HiGHS/pyomo deps declared in tests/BUILD)
uv run ruff format .          # format
uv run ruff check --fix .     # lint and auto-fix
uv run pyrefly check          # type-check
```

`uv run ...` automatically syncs the project `.venv` from `pyproject.toml`
before invoking the tool, so pyrefly can always resolve `pyomo`, `highspy`,
`pytest`, etc. No manual `pip install` step is needed; on a fresh checkout
the first `uv run` call provisions everything.

## Environment notes

- Pants pulls runtime deps into its own pex venvs under
  `~/.cache/pants/named_caches/pex_root/`. The default cache location works
  fine — no need to redirect it into the working directory.
- Pyrefly config lives in `pyproject.toml` under `[tool.pyrefly]` and adds
  `stubs/` to the search path so the local `pyomo.environ` Protocol stubs in
  `stubs/pyomo/environ.pyi` are picked up.

## Commit checklist

Before `git commit`:

1. `pants test ::` is green and shows **100%** coverage across `src/` and
   `tests/`.
2. `uv run ruff format .` reports `N files left unchanged` (no further
   reformatting needed).
3. `uv run ruff check --fix .` reports `All checks passed!`.
4. `uv run pyrefly check` reports `0 errors`.
