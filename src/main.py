import sys

import pyomo.environ as pyo
from solvers_list import solvers_list


def main() -> int:
    """Calculate the optimal food mix"""
    print(f"Available solvers: {solvers_list()}")

    model = pyo.ConcreteModel()
    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
    model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])
    model.Constraint1 = pyo.Constraint(
        expr=3 * model.x[1] + 4 * model.x[2] >= 1
    )

    opt = pyo.SolverFactory("highs")
    if not opt.available():
        print("Solver 'highs' is not available.")
        return 1

    results = opt.solve(model)
    pyo.assert_optimal_termination(results)

    print(
        f"Objective value: {pyo.value(model.OBJ)} x = {[pyo.value(v) for v in model.x]}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
