import dataclasses

import pytest
from display_units import DisplayUnits


def test_continuous_round_trips_attributes() -> None:
    du = DisplayUnits(
        singular="g",
        plural="g",
        per_labeled_serving=78.0,
        per_optimization_unit=None,
    )
    assert du.singular == "g"
    assert du.plural == "g"
    assert du.per_labeled_serving == 78.0
    assert du.per_optimization_unit is None


def test_discrete_half_pill_valid() -> None:
    du = DisplayUnits(
        singular="pill",
        plural="pills",
        per_labeled_serving=4.0,
        per_optimization_unit=0.5,
    )
    assert du.per_optimization_unit is not None
    assert du.per_labeled_serving / du.per_optimization_unit == 8.0


def test_discrete_whole_pill_valid() -> None:
    du = DisplayUnits(
        singular="pill",
        plural="pills",
        per_labeled_serving=1.0,
        per_optimization_unit=1.0,
    )
    assert du.per_optimization_unit == 1.0


def test_discrete_invalid_ratio_raises() -> None:
    with pytest.raises(ValueError, match="not an integer multiple"):
        DisplayUnits(
            singular="pill",
            plural="pills",
            per_labeled_serving=4.0,
            per_optimization_unit=0.3,
        )


def test_isclose_tolerance_allows_near_integer_ratio() -> None:
    DisplayUnits(
        singular="pill",
        plural="pills",
        per_labeled_serving=2.0,
        per_optimization_unit=2.0 - 1e-12,
    )


def test_frozen_dataclass_rejects_mutation() -> None:
    du = DisplayUnits(
        singular="g",
        plural="g",
        per_labeled_serving=90.0,
        per_optimization_unit=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        du.singular = "kg"  # type: ignore[misc]
