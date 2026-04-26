import math

import pytest
from display_units import DisplayUnits
from food import Cost, Food, NutritionFacts


def _make_nutrition(
    calories: float = 100.0,
    serving_size: float = 10.0,
    total_fat: float = 0.0,
    protein: float = 0.0,
) -> NutritionFacts:
    return NutritionFacts(
        serving_size=serving_size,
        calories=calories,
        total_fat=total_fat,
        saturated_fat=0.0,
        trans_fat=0.0,
        cholesterol=0.0,
        sodium=0.0,
        total_carbohydrate=0.0,
        dietary_fiber=None,
        total_sugars=None,
        added_sugars=None,
        protein=protein,
    )


def _continuous_food() -> Food:
    return Food(
        name="Test Powder",
        short_name="TP",
        cost=Cost(grams_per_package=100.0, cents_per_package=100.0),
        nutrition_facts=_make_nutrition(serving_size=10.0),
        display_units=DisplayUnits(
            singular="g",
            plural="g",
            per_labeled_serving=10.0,
            per_optimization_unit=None,
        ),
    )


def _discrete_food() -> Food:
    return Food(
        name="Test Pills",
        short_name="TPi",
        cost=Cost(grams_per_package=100.0, cents_per_package=400.0),
        nutrition_facts=_make_nutrition(serving_size=10.0),
        display_units=DisplayUnits(
            singular="pill",
            plural="pills",
            per_labeled_serving=4.0,
            per_optimization_unit=0.5,
        ),
    )


def test_labeled_servings_per_optimization_unit_continuous() -> None:
    food = _continuous_food()
    # 1 opt unit = 1 g; serving = 10 g → 0.1 servings/opt unit
    assert food.labeled_servings_per_optimization_unit() == pytest.approx(0.1)


def test_labeled_servings_per_optimization_unit_discrete() -> None:
    food = _discrete_food()
    # 0.5 pill / 4 pills = 0.125 servings per opt unit
    assert food.labeled_servings_per_optimization_unit() == pytest.approx(0.125)


def test_dollars_per_optimization_unit_continuous() -> None:
    food = _continuous_food()
    # 100 g / pkg ÷ 10 g/serving = 10 servings/pkg
    # 100 cents / 10 servings = 10 c/serving = $0.10/serving
    # × 0.1 servings/opt unit = $0.01/opt unit
    assert food.dollars_per_optimization_unit() == pytest.approx(0.01)


def test_dollars_per_optimization_unit_discrete() -> None:
    food = _discrete_food()
    # 10 servings/pkg, 400 c/pkg → 40 c/serving = $0.40/serving
    # × 0.125 servings/opt unit = $0.05/opt unit
    assert food.dollars_per_optimization_unit() == pytest.approx(0.05)


def test_effective_carbohydrates_positive() -> None:
    food = Food(
        name="Carb Stuff",
        short_name="CS",
        cost=Cost(grams_per_package=100.0, cents_per_package=100.0),
        nutrition_facts=_make_nutrition(
            calories=100.0,
            total_fat=2.0,  # 18 cal
            protein=4.0,  # 16 cal
        ),
        display_units=DisplayUnits(
            singular="g",
            plural="g",
            per_labeled_serving=10.0,
            per_optimization_unit=None,
        ),
    )
    # (100 - 18 - 16) / 4 = 16.5
    assert food.effective_carbohydrates() == pytest.approx(16.5)


def test_effective_carbohydrates_clamped_to_zero() -> None:
    food = Food(
        name="Pure Fat Bomb",
        short_name="PFB",
        cost=Cost(grams_per_package=100.0, cents_per_package=100.0),
        nutrition_facts=_make_nutrition(
            calories=100.0,
            total_fat=20.0,  # 180 cal — exceeds total
            protein=0.0,
        ),
        display_units=DisplayUnits(
            singular="g",
            plural="g",
            per_labeled_serving=10.0,
            per_optimization_unit=None,
        ),
    )
    assert food.effective_carbohydrates() == 0.0


def test_continuous_treats_per_opt_unit_as_one_display_unit() -> None:
    food = _continuous_food()
    # Sanity: continuous mode is equivalent to (per_optimization_unit=1.0,
    # but real-valued LP variable). Ratio should be 1/per_labeled_serving.
    assert math.isclose(
        food.labeled_servings_per_optimization_unit(),
        1.0 / food.display_units.per_labeled_serving,
    )
