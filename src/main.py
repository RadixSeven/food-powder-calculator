import sys

import pyomo.environ as pyo
from food import Food
from gs_whey import gs_whey
from hlth_code import hlth_code
from soylent import soylent_1_9

foods = [soylent_1_9, hlth_code, gs_whey]


def cal(food: Food) -> str:
    """Return the variable name for the calories of a food in the daily mix."""
    return f"cal {food.short_name}"


def main() -> int:
    """Calculate the optimal food mix"""
    max_carbs = 80  # Maximum carbohydrates per day in grams

    model = pyo.ConcreteModel()
    model.name = "Optimal Food Mix"
    # Fraction of the food by calories in the final mix
    calories_per_day = 2000
    model.cal_food = pyo.Var(
        [cal(food) for food in foods],
        domain=pyo.NonNegativeReals,
    )
    model.total_frac = pyo.Constraint(
        expr=sum(model.cal_food[cal(food)] for food in foods)
        == calories_per_day
    )
    model.OBJ = pyo.Objective(
        expr=sum(
            food.dollars_per_calorie() * model.cal_food[cal(food)]
            for food in foods
        ),
        sense=pyo.minimize,
    )
    model.carbs = pyo.Constraint(
        expr=sum(
            food.nutrition_facts.total_carbohydrate
            * model.cal_food[cal(food)]
            * food.servings_per_calorie()
            for food in foods
        )
        <= max_carbs,
    )

    # model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
    # model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])
    # model.Constraint1 = pyo.Constraint(
    #     expr=3 * model.x[1] + 4 * model.x[2] >= 1
    # )

    opt = pyo.SolverFactory("highs")
    if not opt.available():
        print("Solver 'highs' is not available.")
        return 1

    results = opt.solve(model)
    pyo.assert_optimal_termination(results)

    print(
        f"Objective value: {pyo.value(model.OBJ)} x = {[pyo.value(v) for v in model.cal_food]}",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
