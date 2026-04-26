# Overview

This is a utility for handing powder mixing for my own personal use. It
provided a chance to practice using the `pyomo` library and deal with the
yak shaving it requires.

The following instructions are intended for my future self.

# Development Installation

```shell
pip install .  # I should separate out dev dependencies
pre-commit install
```

# Running

```shell
pants run src:main # I had a "better" name but this was easier to remember
```

# Testing and static analysis

```shell
pants check :: && pants lint :: && pants test ::
uvx ruff format .          # format
uvx ruff check --fix .     # lint and auto-fix
uvx pyrefly check          # type-check (config under [tool.pyrefly] in pyproject.toml)
```

Before committing, all tests and static analysis must pass, tests must have
100% coverage, and the tree must be fully formatted.
