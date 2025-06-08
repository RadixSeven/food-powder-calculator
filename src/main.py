import contextlib
import os
import sys

import pyomo.environ as pyo


def is_valid_solver_string(solver_name: str) -> bool:
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
    return [
        s
        for s in pyo.SolverFactory.__dict__["_cls"].keys()
        if is_valid_solver_string(s)
    ]


def main() -> int:
    """Calculate the optimal food mix"""
    print(f"Available solvers: {solvers_list()}")

    model = pyo.ConcreteModel()
    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
    model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])
    model.Constraint1 = pyo.Constraint(
        expr=3 * model.x[1] + 4 * model.x[2] >= 1
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
