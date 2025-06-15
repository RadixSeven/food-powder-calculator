from food import Food, Cost, NutritionFacts


gs_whey = Food(
    name=(
        "Optimum Nutrition Gold Standard 100% Whey"
        "Vanilla Ice Cream Protein Powder Drink Mix"
    ),
    short_name="Gold Standard Whey",
    cost=Cost(
        grams_per_package=2263.0,
        cents_per_package=6899.0,  # 68.99 for a 5 lb package
    ),
    nutrition_facts=NutritionFacts(
        serving_size=31.0,
        calories=120.0,
        total_fat=1.5,
        saturated_fat=1,
        trans_fat=None,
        cholesterol=55.0,
        sodium=130.0,
        total_carbohydrate=4.0,
        dietary_fiber=None,
        total_sugars=1.0,
        added_sugars=None,
        protein=24.0,
        calcium=10.0,
        potassium=4.0,
    ),
)
