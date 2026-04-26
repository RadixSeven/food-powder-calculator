import io
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import patch

import pyomo.environ as pyo
import pytest
from main import foods, get_per_day, main, units


def test_main_runs_and_returns_zero() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main()
    assert rc == 0
    out = buf.getvalue()
    # Header lines from the optimizer output
    assert "Objective:" in out
    assert "Cost: $" in out
    assert "4 Days' Food mix:" in out
    # Sanity: total calories constraint is tight
    assert "Actual Calories: 2000 per day" in out


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
