from food import Food, Cost, NutritionFacts

soylent_1_9 = Food(
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
