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
    # Get at least 100% of the daily recommended intake of vitamin D
    model.vitamin_d = pyo.Constraint(
        expr=sum(
            food.nutrition_facts.vitamin_d
            * model.cal_food[cal(food)]
            * food.servings_per_calorie()
            for food in foods
            if food.nutrition_facts.vitamin_d is not None
        )
        >= 100,
    )
    # Get at least 100% of the fiber RDI (30 g for males over 50)
    model.fiber = pyo.Constraint(
        expr=sum(
            food.nutrition_facts.dietary_fiber
            * model.cal_food[cal(food)]
            * food.servings_per_calorie()
            for food in foods
            if food.nutrition_facts.dietary_fiber is not None
        )
        >= 30,
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
                f"{food.short_name:>20}: {c:4.0f} calories ({grams:3.0f} grams)",
            )
    print("")

    # We need calculated calories to satisfy the FitBit food app which
    # expects consistent calories from the macronutrients.
    calculated_calories = (
        round(get_per_day("protein", model)) * 4
        + round(get_per_day("total_fat", model)) * 9
        + round(get_per_day("effective_carbohydrates", model)) * 4
    )
    print(f"{'Calculated Calories':>20}: {calculated_calories:4.0f} per day")

    for field, field_name in [
        ("calories", "Actual Calories"),
        ("total_fat", "Fat (g)"),
        ("cholesterol", "Cholesterol (mg)"),
        ("sodium", "Sodium (mg)"),
        ("effective_carbohydrates", "Carbohydrates (g)"),
        ("dietary_fiber", "Fiber (g)"),
        ("total_sugars", "Sugar (g)"),
        ("protein", "Protein (g)"),
        ("vitamin_d", "Vitamin D (%)"),
    ]:
        per_day = get_per_day(field, model)
        print(f"{field_name:>20}: {per_day:4.0f} per day")

    return 0


def get_per_day(field: str, model: pyo.ConcreteModel) -> float:
    """Get the per-day value of a field from the model."""

    def field_value(food: Food) -> float:
        """Get the value of a field from the food's nutrition facts."""
        if field == "effective_carbohydrates":
            return food.effective_carbohydrates()
        a = getattr(food.nutrition_facts, field)
        if a is None:
            return 0
        if isinstance(a, float):
            return a
        raise ValueError(f"Unexpected type for field {field}: {type(a)}")

    return sum(
        field_value(food)
        * float(pyo.value(model.cal_food[cal(food)]))
        * food.servings_per_calorie()
        for food in foods
    )


if __name__ == "__main__":
    sys.exit(main())
