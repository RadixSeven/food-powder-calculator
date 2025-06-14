import sys
from dataclasses import dataclass

import pyomo.environ as pyo


@dataclass
class Cost:
    """Dataclass to hold cost information for a food item.

    Attributes:
        cents_per_package: one package's cost in cents
        grams_per_package: the weight of one package's food in grams
    """

    cents_per_package: float
    grams_per_package: float


@dataclass
class NutritionFacts:
    """Dataclass to hold nutrition facts for a food item.

    Micronutrients are given in percentage of daily value (DV),
     based on a 2000-calorie diet. 0-100% DV is represented as 0.0-100.0.

    Attributes:
        serving_size: the size of the serving in grams
        total_fat: the total fat content in grams
        saturated_fat: the saturated fat content in grams
        trans_fat: the trans fat content in grams
        cholesterol: the cholesterol content in milligrams
        sodium: the sodium content in milligrams
        total_carbohydrate: the total carbohydrate content in grams
        dietary_fiber: the dietary fiber content in grams, or None if not listed
        total_sugars: the total sugars content in grams, or None if not listed
        added_sugars: the added sugars content in grams, or None if not listed
        protein: the protein content in grams
    """

    serving_size: float
    total_fat: float
    saturated_fat: float
    trans_fat: float
    cholesterol: float
    sodium: float
    total_carbohydrate: float
    dietary_fiber: float | None
    total_sugars: float | None
    added_sugars: float | None
    protein: float
    vitamin_d: float | None = None
    calcium: float | None = None
    iron: float | None = None
    potassium: float | None = None
    vitamin_a: float | None = None
    vitamin_c: float | None = None
    vitamin_e: float | None = None
    vitamin_k: float | None = None
    thiamine: float | None = None
    riboflavin: float | None = None
    niacin: float | None = None
    vitamin_b6: float | None = None
    folate: float | None = None
    vitamin_b12: float | None = None
    biotin: float | None = None
    pantothenic_acid: float | None = None
    phosphorus: float | None = None
    iodine: float | None = None
    magnesium: float | None = None
    zinc: float | None = None
    selenium: float | None = None
    copper: float | None = None
    manganese: float | None = None
    chromium: float | None = None
    molybdenum: float | None = None
    chloride: float | None = None
    choline: float | None = None


@dataclass
class Food:
    """Dataclass to hold information about a food item.

    Attributes:
        name: the name of the food item
        short_name: a short name or identifier for the food item
        cost: the cost of the food item
        nutrition_facts: the nutrition facts of the food item
    """

    name: str
    short_name: str
    cost: Cost
    nutrition_facts: NutritionFacts


foods = [
    Food(
        name=(
            "Soylent Original Naturally Flavored Powdered Food Complete "
            "Meal Formula v1.9"
        ),
        short_name="Soylent 1.9",
        cost=Cost(
            grams_per_package=450.0,
            cents_per_package=12300.0 / 14,  # $123.00 for 14 packages
        ),
        nutrition_facts=NutritionFacts(
            serving_size=90.0,
            total_fat=19.0,
            saturated_fat=2.0,
            trans_fat=0.0,
            cholesterol=0.0,
            sodium=320.0,
            total_carbohydrate=42.0,
            dietary_fiber=6.0,
            total_sugars=16.0,
            added_sugars=15.0,
            protein=20.0,
            vitamin_d=20.0,
            calcium=20.0,
            iron=20.0,
            potassium=20.0,
            vitamin_a=20.0,
            vitamin_c=20.0,
            vitamin_e=20.0,
            vitamin_k=20.0,
            thiamine=20.0,
            riboflavin=20.0,
            niacin=20.0,
            vitamin_b6=20.0,
            folate=20.0,
            vitamin_b12=20.0,
            biotin=20.0,
            pantothenic_acid=20.0,
            phosphorus=20.0,
            iodine=20.0,
            magnesium=20.0,
            zinc=20.0,
            selenium=20.0,
            copper=20.0,
            manganese=20.0,
            chromium=20.0,
            molybdenum=20.0,
            chloride=20.0,
            choline=20.0,
        ),
    )
]


def main() -> int:
    """Calculate the optimal food mix"""
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
