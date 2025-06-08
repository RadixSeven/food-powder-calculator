"""Module to list available solvers for Pyomo."""

import contextlib
import os

from pyomo import environ as pyo


def is_valid_solver_string(solver_name: str) -> bool:
    """Return True if the solver_name is a valid Pyomo solver string."""
    if solver_name.startswith("_mock_"):
        return False

    # noinspection PyBroadException
    try:
        with open(os.devnull, "w") as devnull:
            # Redirect stdout to devnull to suppress warnings about missing
            # files or solvers for non-installed solvers.
            with contextlib.redirect_stdout(devnull):
                return bool(pyo.SolverFactory(solver_name).available())
    except Exception:
        return False


def solvers_list() -> list[str]:
    """Return a list of available solvers for Pyomo."""
    return [
        s
        for s in pyo.SolverFactory.__dict__["_cls"].keys()
        if is_valid_solver_string(s)
    ]
