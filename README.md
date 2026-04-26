# Overview

This is a utility for handing powder mixing for my own personal use. It
provided a chance to practice using the `pyomo` library and deal with the
yak shaving it requires.

The following instructions are intended for my future self.

# Development Installation

```shell
uv sync                          # populate .venv from pyproject.toml
uv run pre-commit install        # install the git pre-commit hook
```

`uv sync` is also implicitly run by every `uv run ...` invocation below, so a
fresh checkout can skip straight to the commands in the next sections — but
`pre-commit install` still has to be run once per clone to enable the
git hook.

# Running

```shell
pants run src:main # I had a "better" name but this was easier to remember
```

# Testing and static analysis

```shell
./run_all_qa.sh               # convenience: runs every gate below in order
```

Or run them individually:

```shell
pants check :: && pants lint :: && pants test ::
uv run ruff format .          # format Python
uv run mdformat .             # format Markdown
uv run ruff check --fix .     # lint Python and auto-fix
uv run pyrefly check          # type-check (config under [tool.pyrefly] in pyproject.toml)
```

The `uv run` invocations auto-sync `.venv` so that pyrefly can resolve
imports of `pyomo`, `highspy`, `pytest`, etc. — no manual `pip install` step
is needed.

Before committing, all tests and static analysis must pass, tests must have
100% coverage, and the tree must be fully formatted.
