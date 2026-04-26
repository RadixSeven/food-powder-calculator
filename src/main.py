import sys

import pyomo.environ as pyo
from food import Food
from gs_whey import gs_whey

from hlth_code import hlth_code
from ht_psyllium_husk import ht_psyllium_husk
from optifiber import optifiber
from soylent import soylent_1_9

foods = [
    soylent_1_9,
    optifiber,
    ht_psyllium_husk,
    hlth_code,
    gs_whey,
]


def units(food: Food) -> str:
    """Return the variable name for the optimization-unit count of a food."""
    return f"units {food.short_name}"


def main() -> int:
    """Calculate the optimal food mix"""
    max_carbs = 80  # Maximum carbohydrates per day in grams
    calories_per_day = 2000  # Calories per day

    model = pyo.ConcreteModel()
    model.name = "Optimal Food Mix"
    # Optimization-unit count of each food in the daily mix.
    model.units = pyo.Var(
        [units(food) for food in foods],
        domain=pyo.NonNegativeReals,
    )
    model.total_calories = pyo.Constraint(
        expr=sum(
            food.nutrition_facts.calories
            * food.labeled_servings_per_optimization_unit()
            * model.units[units(food)]
            for food in foods
        )
        == calories_per_day
    )
    # Roughly minimize saturated fat and then cost
    model.OBJ = pyo.Objective(
        expr=sum(
            food.dollars_per_optimization_unit() * model.units[units(food)]
            for food in foods
        )
        + 100
        * sum(
            food.nutrition_facts.saturated_fat
            * food.labeled_servings_per_optimization_unit()
            * model.units[units(food)]
            for food in foods
            if food.nutrition_facts.saturated_fat is not None
        ),
        sense=pyo.minimize,
    )
    # Constrain carbs
    model.carbs = pyo.Constraint(
        expr=sum(
            food.effective_carbohydrates()
            * food.labeled_servings_per_optimization_unit()
            * model.units[units(food)]
            for food in foods
        )
        <= max_carbs,
    )
    # Use different sources of fiber - equal contribution
    # from each source
    optifiber_fiber = optifiber.nutrition_facts.dietary_fiber
    psyllium_fiber = ht_psyllium_husk.nutrition_facts.dietary_fiber
    assert optifiber_fiber is not None
    assert psyllium_fiber is not None
    model.equal_fiber_contrib = pyo.Constraint(
        expr=(
            optifiber_fiber
            * optifiber.labeled_servings_per_optimization_unit()
            * model.units[units(optifiber)]
        )
        - (
            psyllium_fiber
            * ht_psyllium_husk.labeled_servings_per_optimization_unit()
            * model.units[units(ht_psyllium_husk)]
        )
        == 0,
    )

    # Get at least 100% of the daily recommended intake of vitamin D
    model.vitamin_d = pyo.Constraint(
        expr=sum(
            food.nutrition_facts.vitamin_d
            * food.labeled_servings_per_optimization_unit()
            * model.units[units(food)]
            for food in foods
            if food.nutrition_facts.vitamin_d is not None
        )
        >= 100,
    )

    # Get at least 100% of the fiber RDI (30 g for males over 50)
    model.fiber = pyo.Constraint(
        expr=sum(
            food.nutrition_facts.dietary_fiber
            * food.labeled_servings_per_optimization_unit()
            * model.units[units(food)]
            for food in foods
            if food.nutrition_facts.dietary_fiber is not None
        )
        >= 30,
    )

    opt = pyo.SolverFactory("highs")
    if not opt.available():  # pragma: no cover
        print("Solver 'highs' is not available.")
        return 1

    results = opt.solve(model)
    pyo.assert_optimal_termination(results)

    print(f"Objective: {pyo.value(model.OBJ):.2f}")
    opt_cost = pyo.value(
        sum(
            food.dollars_per_optimization_unit() * model.units[units(food)]
            for food in foods
        )
    )
    print(
        f"Cost: ${opt_cost:.2f}/day",
    )
    num_days = 4
    print(f"{num_days} Days' Food mix:")
    for food in foods:
        u = model.units[units(food)]
        opt_units_value = num_days * pyo.value(u)
        if opt_units_value > 0:
            calories = (
                opt_units_value
                * food.labeled_servings_per_optimization_unit()
                * food.nutrition_facts.calories
            )
            grams = (
                opt_units_value
                * food.labeled_servings_per_optimization_unit()
                * food.nutrition_facts.serving_size
            )
            print(
                f"{food.short_name:>20}: "
                f"{calories:4.0f} calories ({grams:3.0f} grams)",
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
        ("saturated_fat", "Sat. Fat (g)"),
        ("trans_fat", "Trans Fat (g)"),
        ("cholesterol", "Cholesterol (mg)"),
        ("sodium", "Sodium (mg)"),
        ("effective_carbohydrates", "Carbohydrates (g)"),
        ("dietary_fiber", "Fiber (g)"),
        ("total_sugars", "Sugars (g)"),
        ("added_sugars", "Added Sugars (g)"),
        ("protein", "Protein (g)"),
        ("vitamin_d", "Vitamin D (%)"),
        ("calcium", "Calcium (%)"),
        ("iron", "Iron (%)"),
        ("potassium", "Potassium (%)"),
        ("thiamine", "B1 Thiamine (%)"),
        ("riboflavin", "B2 Riboflavin (%)"),
        ("niacin", "B3 Niacin (%)"),
        ("pantothenic_acid", "B5 Pantothenic Acid (%)"),
        ("vitamin_b6", "B6 Pyridoxine (%)"),
        ("vitamin_b12", "B12 Cobalamin (%)"),
        ("biotin", "Biotin (%)"),
        ("choline", "Choline (%)"),
        ("folate", "Folate (%)"),
        ("vitamin_a", "Vitamin A (%)"),
        ("vitamin_c", "Vitamin C (%)"),
        ("vitamin_e", "Vitamin E (%)"),
        ("vitamin_k", "Vitamin K (%)"),
        ("chromium", "Chromium (%)"),
        ("copper", "Copper (%)"),
        ("iodine", "Iodine (%)"),
        ("magnesium", "Magnesium (%)"),
        ("manganese", "Manganese (%)"),
        ("molybdenum", "Molybdenum (%)"),
        ("phosphorus", "Phosphorus (%)"),
        ("selenium", "Selenium (%)"),
        ("zinc", "Zinc (%)"),
    ]:
        per_day = get_per_day(field, model)
        print(f"{field_name:>24}: {per_day:4.0f} per day")

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
        if isinstance(a, int):
            if -9007199254740991 < a < 9007199254740991:
                return float(a)
            raise ValueError(
                f"Integer value for field {field} is too large to fit "
                f"exactly in a float: {a}",
            )
        raise ValueError(f"Unexpected type for field {field}: {type(a)}")

    return sum(
        field_value(food)
        * food.labeled_servings_per_optimization_unit()
        * float(pyo.value(model.units[units(food)]))
        for food in foods
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
