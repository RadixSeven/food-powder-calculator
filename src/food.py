from dataclasses import dataclass

from display_units import DisplayUnits


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
        calories: the number of calories per serving
        total_fat: the total fat content in grams
        saturated_fat: the saturated fat content in grams
        trans_fat: the trans fat content in grams or None if not listed
        cholesterol: the cholesterol content in milligrams
        sodium: the sodium content in milligrams
        total_carbohydrate: the total carbohydrate content in grams
        dietary_fiber: the dietary fiber content in grams, or None if not listed
        total_sugars: the total sugars content in grams, or None if not listed
        added_sugars: the added sugars content in grams, or None if not listed
        protein: the protein content in grams
    """

    serving_size: float
    calories: float
    total_fat: float
    saturated_fat: float
    trans_fat: float | None
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
        display_units: how to display and discretize this food's quantity
    """

    name: str
    short_name: str
    cost: Cost
    nutrition_facts: NutritionFacts
    display_units: DisplayUnits

    def labeled_servings_per_optimization_unit(self) -> float:
        """Return the number of labeled servings represented by one optimization unit.

        For continuous foods (``per_optimization_unit is None``) we treat one
        optimization unit as one display unit, so the conversion is
        ``1 / per_labeled_serving``. For discrete foods the conversion is
        ``per_optimization_unit / per_labeled_serving``.
        """
        per_opt = self.display_units.per_optimization_unit
        display_units_per_opt = 1.0 if per_opt is None else per_opt
        return display_units_per_opt / self.display_units.per_labeled_serving

    def dollars_per_optimization_unit(self) -> float:
        """Calculate the cost of one optimization unit in dollars."""
        servings_per_package = (
            self.cost.grams_per_package / self.nutrition_facts.serving_size
        )
        cents_per_serving = self.cost.cents_per_package / servings_per_package
        dollars_per_serving = cents_per_serving / 100.0
        return (
            dollars_per_serving * self.labeled_servings_per_optimization_unit()
        )

    def effective_carbohydrates(self) -> float:
        """Calculate the effective carbohydrates per serving.

        This subtracts other caloric sources from total calories and
        then divides by the calories per gram of carbohydrate (4 kcal/g) to
        get the effective digestible carbohydrates.
        """
        total_calories = self.nutrition_facts.calories
        fat_calories = self.nutrition_facts.total_fat * 9.0
        protein_calories = self.nutrition_facts.protein * 4.0
        carb_calories = total_calories - fat_calories - protein_calories
        if carb_calories < 0:
            return 0.0
        return carb_calories / 4.0
