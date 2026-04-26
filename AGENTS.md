# Agent instructions

## Tests and static analysis

Easiest: run `./run_all_qa.sh`, which runs every gate in order and prints a
single green/red verdict to stderr. Reformatting alone is not a failure.

If you want to invoke them individually:

```shell
pants test ::                 # tests + coverage (HiGHS/pyomo deps declared in tests/BUILD)
uv run ruff format .          # format Python
uv run mdformat .             # format Markdown
uv run ruff check --fix .     # lint Python and auto-fix
uv run pyrefly check          # type-check
uvx --from shellcheck-py shellcheck --severity=style --enable=all run_all_qa.sh
```

Everything must pass before committing, tests must have 100% coverage, and
the tree must be fully formatted.

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

1. `./run_all_qa.sh` exits 0 and prints "All quality gates passed".
1. `pants test ::` is green and shows **100%** coverage across `src/` and
   `tests/`.
1. `uv run ruff format .` and `uv run mdformat .` both report no remaining
   reformatting.
1. `uv run ruff check --fix .` reports `All checks passed!`.
1. `uv run pyrefly check` reports `0 errors`.
