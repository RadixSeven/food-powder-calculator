import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DisplayUnits:
    """How a food or supplement's quantity is displayed and discretized.

    Bridges three quantity scales used by the optimizer:

    * The optimization variable's value (an integer if discrete, a real if not).
    * The display quantity shown to the user, e.g. ``"2 pills"`` or ``"78 g"``.
    * The number of labeled servings as defined on the product's nutrition label.

    The optimization unit is a synthetic discretization choice: the smallest
    amount of the item that can practically be prepared. For halvable pills it
    is half a pill; for powders measured in grams it is undefined and the LP
    variable is continuous.

    Attributes:
        singular: Display-unit name in singular form, e.g. ``"pill"``, ``"g"``.
        plural: Display-unit name in plural form, e.g. ``"pills"``, ``"g"``.
        per_labeled_serving: Display units in one labeled serving. ``78.0`` for
            a powder whose label reads "Serving size: 78 g"; ``4.0`` for a
            supplement whose label reads "Serving size: 4 capsules".
        per_optimization_unit: Display units per optimization unit, or ``None``
            for continuous variables. ``0.5`` for half-divisible pills,
            ``1.0`` for whole-pill-only items, ``None`` for powders.
    """

    singular: str
    plural: str
    per_labeled_serving: float
    per_optimization_unit: float | None

    def __post_init__(self) -> None:
        """Validate that a labeled serving is an exact multiple of optimization units.

        Raises:
            ValueError: If ``per_optimization_unit`` is set and
                ``per_labeled_serving`` is not an integer multiple of it. A
                labeled serving must be exactly representable by some whole
                number of optimization units, otherwise the solver cannot
                produce solutions expressed in whole servings.
        """
        if self.per_optimization_unit is None:
            return
        ratio = self.per_labeled_serving / self.per_optimization_unit
        if not math.isclose(ratio, round(ratio)):
            raise ValueError(
                f"A labeled serving ({self.per_labeled_serving} "
                f"{self.plural}) is not an integer multiple of the "
                f"optimization unit ({self.per_optimization_unit} "
                f"{self.plural})."
            )