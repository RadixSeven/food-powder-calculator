from food import Food, Cost, NutritionFacts

hlth_code = Food(
    name=(
        "HLTH Code Complete Meal Nutritional Shake made with "
        "Real Ingredients Creamy Vanilla"
    ),
    short_name="HLTH Vanilla",
    cost=Cost(
        grams_per_package=1170.0,
        cents_per_package=9990.0 / 2,  # 99.90 for 2 packages
    ),
    nutrition_facts=NutritionFacts(
        serving_size=78.0,
        total_fat=27.0,
        saturated_fat=17.0,
        trans_fat=0.1,
        cholesterol=45.0,
        sodium=163.0,
        total_carbohydrate=13.0,
        dietary_fiber=9.0,
        total_sugars=2.0,
        added_sugars=0.0,
        protein=27.0,
        vitamin_d=50.0,
        calcium=31.0,
        iron=50.0,
        potassium=9.0,
        vitamin_a=50.0,
        vitamin_c=50.0,
        vitamin_e=50.0,
        vitamin_k=50.0,
        thiamine=50.0,
        riboflavin=50.0,
        niacin=50.0,
        vitamin_b6=50.0,
        folate=50.0,
        vitamin_b12=50.0,
        biotin=50.0,
        pantothenic_acid=50.0,
        phosphorus=50.0,
        iodine=50.0,
        magnesium=50.0,
        zinc=50.0,
        selenium=50.0,
        copper=50.0,
        manganese=50.0,
        chromium=50.0,
        molybdenum=50.0,
    ),
)
