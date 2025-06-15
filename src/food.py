from dataclasses import dataclass


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
    """

    name: str
    short_name: str
    cost: Cost
    nutrition_facts: NutritionFacts

    def dollars_per_calorie(self) -> float:
        """Calculate the cost of the food item per calorie."""
        servings_per_package = (
            self.cost.grams_per_package / self.nutrition_facts.serving_size
        )
        cents_per_serving = self.cost.cents_per_package / servings_per_package
        cents_per_calorie = cents_per_serving / self.nutrition_facts.calories
        return cents_per_calorie / 100.0

    def servings_per_calorie(self) -> float:
        """Calculate the servings of food per calorie."""
        return 1 / self.nutrition_facts.calories
