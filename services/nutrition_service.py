from dataclasses import dataclass

# --- Data Classes ---

@dataclass
class NutritionRequest:
    age: int
    weight_kg: float
    height_cm: float
    gender: str            # "Male" or "Female"
    activity_level: str    # "sedentary", "light", "moderate", "active", "very_active"
    goal: str              # "weight_loss", "muscle_gain", "maintenance", "endurance"


@dataclass
class MacroBreakdown:
    protein_g: float
    carbs_g: float
    fats_g: float


@dataclass
class NutritionResult:
    bmr: float
    tdee: float
    target_calories: float
    macros: MacroBreakdown
    goal: str
    meal_structure: list[dict]
    supplement_suggestions: list[str]


# --- Activity Multipliers (Harris-Benedict standard) ---
ACTIVITY_MULTIPLIERS = {
    "sedentary":   1.2,    # desk job, no exercise
    "light":       1.375,  # light exercise 1-3 days/week
    "moderate":    1.55,   # moderate exercise 3-5 days/week
    "active":      1.725,  # hard exercise 6-7 days/week
    "very_active": 1.9     # physical job + hard exercise
}

# --- Calorie Adjustments per Goal ---
CALORIE_ADJUSTMENTS = {
    "weight_loss":  -500,   # 500 kcal deficit → ~0.5kg/week loss
    "muscle_gain":  +300,   # 300 kcal surplus → lean bulk
    "maintenance":     0,   # maintain current weight
    "endurance":    +100    # slight surplus for performance
}

# --- Macro Ratios per Goal (protein%, carbs%, fats%) ---
MACRO_RATIOS = {
    "weight_loss":  (0.40, 0.35, 0.25),  # high protein to preserve muscle
    "muscle_gain":  (0.30, 0.50, 0.20),  # high carbs for energy and growth
    "maintenance":  (0.30, 0.40, 0.30),  # balanced
    "endurance":    (0.25, 0.55, 0.20)   # high carbs for sustained energy
}

# --- Meal Structure Templates per Goal ---
MEAL_TEMPLATES = {
    "weight_loss": [
        {"meal": "Breakfast", "timing": "7:00 AM", "calorie_share": 0.25, "focus": "High protein, low carb"},
        {"meal": "Snack",     "timing": "10:00 AM","calorie_share": 0.10, "focus": "Protein snack"},
        {"meal": "Lunch",     "timing": "1:00 PM", "calorie_share": 0.30, "focus": "Lean protein + vegetables"},
        {"meal": "Snack",     "timing": "4:00 PM", "calorie_share": 0.10, "focus": "Low calorie snack"},
        {"meal": "Dinner",    "timing": "7:00 PM", "calorie_share": 0.25, "focus": "Light protein + salad"},
    ],
    "muscle_gain": [
        {"meal": "Breakfast",     "timing": "7:00 AM", "calorie_share": 0.25, "focus": "Protein + complex carbs"},
        {"meal": "Pre-Workout",   "timing": "11:00 AM","calorie_share": 0.15, "focus": "Fast carbs + protein"},
        {"meal": "Lunch",         "timing": "1:00 PM", "calorie_share": 0.25, "focus": "Balanced meal"},
        {"meal": "Post-Workout",  "timing": "4:00 PM", "calorie_share": 0.15, "focus": "Protein + fast carbs"},
        {"meal": "Dinner",        "timing": "7:00 PM", "calorie_share": 0.20, "focus": "Protein + slow carbs"},
    ],
    "maintenance": [
        {"meal": "Breakfast", "timing": "7:00 AM", "calorie_share": 0.25, "focus": "Balanced macros"},
        {"meal": "Snack",     "timing": "10:00 AM","calorie_share": 0.10, "focus": "Light snack"},
        {"meal": "Lunch",     "timing": "1:00 PM", "calorie_share": 0.35, "focus": "Balanced meal"},
        {"meal": "Snack",     "timing": "4:00 PM", "calorie_share": 0.10, "focus": "Light snack"},
        {"meal": "Dinner",    "timing": "7:00 PM", "calorie_share": 0.20, "focus": "Light balanced meal"},
    ],
    "endurance": [
        {"meal": "Breakfast",    "timing": "6:30 AM", "calorie_share": 0.20, "focus": "Complex carbs + protein"},
        {"meal": "Pre-Workout",  "timing": "9:00 AM", "calorie_share": 0.15, "focus": "Fast carbs"},
        {"meal": "Lunch",        "timing": "12:00 PM","calorie_share": 0.25, "focus": "High carb + protein"},
        {"meal": "Post-Workout", "timing": "3:00 PM", "calorie_share": 0.20, "focus": "Carbs + electrolytes"},
        {"meal": "Dinner",       "timing": "7:00 PM", "calorie_share": 0.20, "focus": "Complex carbs + protein"},
    ]
}

# --- Supplement Suggestions per Goal ---
SUPPLEMENT_SUGGESTIONS = {
    "weight_loss": [
        "Whey Protein — preserve muscle during caloric deficit",
        "Green Tea Extract — mild metabolism boost",
        "Fiber Supplement — improves satiety",
    ],
    "muscle_gain": [
        "Whey Protein — post-workout muscle synthesis",
        "Creatine Monohydrate — proven strength and mass gains",
        "Casein Protein — slow release protein before bed",
    ],
    "maintenance": [
        "Multivitamin — cover micronutrient gaps",
        "Omega-3 Fish Oil — anti-inflammatory, heart health",
    ],
    "endurance": [
        "Electrolyte Supplement — sodium, potassium, magnesium",
        "Beta-Alanine — delays muscle fatigue",
        "Carbohydrate Gel — quick energy during long sessions",
    ]
}


class NutritionService:

    def calculate(self, request: NutritionRequest) -> NutritionResult:

        # --- Step 1: BMR (Harris-Benedict Revised by Mifflin-St Jeor 1990) ---
        # More accurate than original 1919 formula
        if request.gender.lower() == "male":
            bmr = (10 * request.weight_kg) + \
                  (6.25 * request.height_cm) - \
                  (5 * request.age) + 5
        else:
            bmr = (10 * request.weight_kg) + \
                  (6.25 * request.height_cm) - \
                  (5 * request.age) - 161

        # --- Step 2: TDEE ---
        multiplier = ACTIVITY_MULTIPLIERS.get(request.activity_level, 1.55)
        tdee = bmr * multiplier

        # --- Step 3: Target Calories ---
        adjustment = CALORIE_ADJUSTMENTS.get(request.goal, 0)
        target_calories = tdee + adjustment

        # Never go below 1200 kcal — minimum safe threshold
        target_calories = max(target_calories, 1200)

        # --- Step 4: Macros ---
        protein_ratio, carbs_ratio, fats_ratio = MACRO_RATIOS.get(
            request.goal, (0.30, 0.40, 0.30)
        )

        # 1g protein = 4 kcal, 1g carbs = 4 kcal, 1g fat = 9 kcal
        macros = MacroBreakdown(
            protein_g=round((target_calories * protein_ratio) / 4, 1),
            carbs_g=round((target_calories * carbs_ratio) / 4, 1),
            fats_g=round((target_calories * fats_ratio) / 9, 1)
        )

        # --- Step 5: Meal Structure ---
        template = MEAL_TEMPLATES.get(request.goal, MEAL_TEMPLATES["maintenance"])
        meal_structure = []
        for meal in template:
            meal_calories = round(target_calories * meal["calorie_share"])
            meal_structure.append({
                "meal": meal["meal"],
                "timing": meal["timing"],
                "calories": meal_calories,
                "focus": meal["focus"]
            })

        # --- Step 6: Supplements ---
        supplements = SUPPLEMENT_SUGGESTIONS.get(request.goal, [])

        return NutritionResult(
            bmr=round(bmr, 1),
            tdee=round(tdee, 1),
            target_calories=round(target_calories, 1),
            macros=macros,
            goal=request.goal,
            meal_structure=meal_structure,
            supplement_suggestions=supplements
        )