import io
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import patch

import pyomo.environ as pyo
import pytest
from display_units import DisplayUnits
from food import Cost, Food, NutritionFacts
from main import (
    foods,
    format_one_day_recipe,
    get_per_day,
    main,
    pluralize,
    units,
)


def test_main_runs_and_returns_zero() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main()
    assert rc == 0
    out = buf.getvalue()
    # Header lines from the optimizer output
    assert "Objective:" in out
    assert "Cost: $" in out
    # The 1-day block precedes the 4-day block.
    assert out.index("1 Day's Food mix:") < out.index("4 Days' Food mix:")
    # All current foods are continuous, so only the Continuous bucket appears.
    assert "Continuous:" in out
    assert "Discrete:" not in out
    # Each food's short_name shows up under the 1-day block followed by "g".
    one_day_block = out.split("1 Day's Food mix:")[1].split(
        "4 Days' Food mix:"
    )[0]
    for food in foods:
        assert food.short_name in one_day_block
    # Sanity: total calories constraint is tight
    assert "Actual Calories: 2000 per day" in out


def test_pluralize_singular_for_one() -> None:
    du = DisplayUnits(
        singular="pill",
        plural="pills",
        per_labeled_serving=1.0,
        per_optimization_unit=1.0,
    )
    assert pluralize(1, du) == "pill"
    assert pluralize(1.0, du) == "pill"


def test_pluralize_plural_for_other_counts() -> None:
    du = DisplayUnits(
        singular="pill",
        plural="pills",
        per_labeled_serving=1.0,
        per_optimization_unit=1.0,
    )
    assert pluralize(2, du) == "pills"
    assert pluralize(0, du) == "pills"
    assert pluralize(1.5, du) == "pills"


def test_pluralize_grams_singular_when_displayed_value_is_one() -> None:
    du = DisplayUnits(
        singular="g",
        plural="g",
        per_labeled_serving=10.0,
        per_optimization_unit=None,
    )
    # Same word for grams either way; still verify the rule fires correctly.
    assert pluralize(1.04, du, precision=0) == "g"
    assert pluralize(1.04, du, precision=1) == "g"  # rounds to 1.0


def test_pluralize_respects_precision_for_continuous_units() -> None:
    du = DisplayUnits(
        singular="scoop",
        plural="scoops",
        per_labeled_serving=1.0,
        per_optimization_unit=None,
    )
    assert pluralize(1.04, du, precision=0) == "scoop"  # rounds to 1
    assert pluralize(1.04, du, precision=1) == "scoop"  # rounds to 1.0
    assert pluralize(1.5, du, precision=1) == "scoops"  # 1.5 != 1


def _continuous_food(short_name: str = "Powder") -> Food:
    return Food(
        name="Test Powder",
        short_name=short_name,
        cost=Cost(grams_per_package=100.0, cents_per_package=100.0),
        nutrition_facts=NutritionFacts(
            serving_size=10.0,
            calories=100.0,
            total_fat=0.0,
            saturated_fat=0.0,
            trans_fat=0.0,
            cholesterol=0.0,
            sodium=0.0,
            total_carbohydrate=0.0,
            dietary_fiber=None,
            total_sugars=None,
            added_sugars=None,
            protein=0.0,
        ),
        display_units=DisplayUnits(
            singular="g",
            plural="g",
            per_labeled_serving=10.0,
            per_optimization_unit=None,
        ),
    )


def _discrete_food(short_name: str = "Pills") -> Food:
    return Food(
        name="Test Pills",
        short_name=short_name,
        cost=Cost(grams_per_package=100.0, cents_per_package=100.0),
        nutrition_facts=NutritionFacts(
            serving_size=10.0,
            calories=100.0,
            total_fat=0.0,
            saturated_fat=0.0,
            trans_fat=0.0,
            cholesterol=0.0,
            sodium=0.0,
            total_carbohydrate=0.0,
            dietary_fiber=None,
            total_sugars=None,
            added_sugars=None,
            protein=0.0,
        ),
        display_units=DisplayUnits(
            singular="pill",
            plural="pills",
            per_labeled_serving=4.0,
            per_optimization_unit=0.5,
        ),
    )


def test_format_one_day_recipe_continuous_only() -> None:
    food = _continuous_food()
    model = pyo.ConcreteModel()
    model.units = pyo.Var(
        [units(food)], domain=pyo.NonNegativeReals, initialize=12.5
    )
    out = format_one_day_recipe(model, [food])
    assert "1 Day's Food mix:" in out
    assert "  Continuous:" in out
    assert "Discrete:" not in out
    assert "Powder" in out and "12.5 g" in out


def test_format_one_day_recipe_discrete_only_with_singular() -> None:
    food = _discrete_food()
    model = pyo.ConcreteModel()
    # 2 opt units × 0.5 pill/opt unit = 1 pill (singular).
    model.units = pyo.Var(
        [units(food)], domain=pyo.NonNegativeReals, initialize=2.0
    )
    out = format_one_day_recipe(model, [food])
    assert "Continuous:" not in out
    assert "  Discrete:" in out
    assert "Pills:" in out
    assert "1 pill" in out
    assert "1 pills" not in out


def test_format_one_day_recipe_discrete_plural() -> None:
    food = _discrete_food()
    model = pyo.ConcreteModel()
    # 6 opt units × 0.5 = 3 pills (plural).
    model.units = pyo.Var(
        [units(food)], domain=pyo.NonNegativeReals, initialize=6.0
    )
    out = format_one_day_recipe(model, [food])
    assert "3 pills" in out


def test_format_one_day_recipe_omits_zero_allocation_foods() -> None:
    food = _continuous_food()
    model = pyo.ConcreteModel()
    model.units = pyo.Var(
        [units(food)], domain=pyo.NonNegativeReals, initialize=0.0
    )
    out = format_one_day_recipe(model, [food])
    # Zero-valued food drops out — both section headers should be absent.
    assert "Continuous:" not in out
    assert "Discrete:" not in out
    assert "Powder" not in out


def test_format_one_day_recipe_mixed() -> None:
    cont = _continuous_food()
    disc = _discrete_food()
    model = pyo.ConcreteModel()
    model.units = pyo.Var(
        [units(cont), units(disc)],
        domain=pyo.NonNegativeReals,
        initialize=1.0,
    )
    out = format_one_day_recipe(model, [cont, disc])
    assert "Continuous:" in out
    assert "Discrete:" in out
    # Continuous block should appear first.
    assert out.index("Continuous:") < out.index("Discrete:")


def test_get_per_day_raises_for_oversize_int() -> None:
    huge = 10**20
    bad_food = replace(
        foods[0],
        nutrition_facts=replace(foods[0].nutrition_facts, calcium=huge),
    )
    with patch("main.foods", [bad_food]):
        with pytest.raises(ValueError, match="too large to fit"):
            get_per_day("calcium", _model_for([bad_food]))


def test_get_per_day_raises_for_unexpected_type() -> None:
    bad_food = replace(
        foods[0],
        nutrition_facts=replace(foods[0].nutrition_facts, calcium="oops"),  # type: ignore[arg-type]
    )
    with patch("main.foods", [bad_food]):
        with pytest.raises(ValueError, match="Unexpected type"):
            get_per_day("calcium", _model_for([bad_food]))


def _model_for(food_list: list) -> pyo.ConcreteModel:  # type: ignore[type-arg]
    model = pyo.ConcreteModel()
    model.units = pyo.Var(
        [units(food) for food in food_list],
        domain=pyo.NonNegativeReals,
        initialize=1,
    )
    return model
