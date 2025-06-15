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
    calories_per_day = 2000  # Calories per day

    model = pyo.ConcreteModel()
    model.name = "Optimal Food Mix"
    # Calories of each food in the final mix
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
            food.effective_carbohydrates()
            * model.cal_food[cal(food)]
            * food.servings_per_calorie()
            for food in foods
        )
        <= max_carbs,
    )

    opt = pyo.SolverFactory("highs")
    if not opt.available():
        print("Solver 'highs' is not available.")
        return 1

    results = opt.solve(model)
    pyo.assert_optimal_termination(results)

    print(
        f"Cost: ${pyo.value(model.OBJ):.2f}/day",
    )
    print("3 Days' Food mix:")
    for food in foods:
        cal_food = model.cal_food[cal(food)]
        c = 3 * pyo.value(cal_food)
        if c > 0:
            grams = (
                c
                * food.servings_per_calorie()
                * food.nutrition_facts.serving_size
            )
            print(
                f"{food.short_name}: {c:.2f} calories ({grams:.2f} grams)",
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
