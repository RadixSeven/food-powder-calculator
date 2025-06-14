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

# Testing

```shell
pants check :: && pants lint :: && pants test ::
```
