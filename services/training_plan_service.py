import joblib
import pandas as pd
import os
from services.exercise_selector import select_exercises

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- Load models once at startup ---
training_plan_model = joblib.load(os.path.join(MODELS_DIR, 'training_plan_model.pkl'))
training_plan_features = joblib.load(os.path.join(MODELS_DIR, 'training_plan_features.pkl'))
training_plan_le = joblib.load(os.path.join(MODELS_DIR, 'training_plan_label_encoder.pkl'))
training_plan_names = joblib.load(os.path.join(MODELS_DIR, 'training_plan_names.pkl'))


def predict_plan_type(profile: dict) -> dict:
    """
    Predicts the user's body type plan based on physical profile.
    Returns plan number and plan type name.
    """
    weight = profile.get('weight_kg', 70)
    height_m = profile.get('height_cm', 170) / 100
    bmi = weight / (height_m ** 2)
    bfp = profile.get('body_fat_percentage') or (bmi * 0.8)
    gender_encoded = 1 if profile.get('gender', 'Male').lower() == 'male' else 0

    bmi_x_bfp = bmi * bfp
    weight_x_bfp = weight * bfp
    bmi_squared = bmi ** 2
    bfp_squared = bfp ** 2
    bmi_x_bfp_squared = bmi_squared * bfp_squared

    if bmi < 18.5:
        bmi_cat = 1
    elif bmi < 24.9:
        bmi_cat = 2
    elif bmi < 29.9:
        bmi_cat = 3
    elif bmi < 34.9:
        bmi_cat = 4
    else:
        bmi_cat = 5

    if bfp < 6:
        bfp_cat = 1
    elif bfp < 14:
        bfp_cat = 2
    elif bfp < 18:
        bfp_cat = 3
    elif bfp < 25:
        bfp_cat = 4
    else:
        bfp_cat = 5

    input_data = pd.DataFrame([{
        'Weight': weight,
        'Height': height_m,
        'BMI': bmi,
        'Body Fat Percentage': bfp,
        'Age': profile.get('age', 25),
        'Gender_Encoded': gender_encoded,
        'BMI_X_BFP': bmi_x_bfp,
        'Weight_X_BFP': weight_x_bfp,
        'BMI_Squared': bmi_squared,
        'BFP_Squared': bfp_squared,
        'BMI_X_BFP_Squared': bmi_x_bfp_squared,
        'BMI_Category_Score': bmi_cat,
        'BFP_Category_Score': bfp_cat,
    }])

    pred_encoded = training_plan_model.predict(input_data[training_plan_features])
    plan_number = training_plan_le.inverse_transform(pred_encoded)[0]
    plan_type = training_plan_names.get(plan_number, 'normal')

    return {
        'plan_number': int(plan_number),
        'plan_type': plan_type
    }


def build_plan(profile: dict, context: dict) -> dict:
    """
    Builds a complete training plan combining ML body type prediction
    and real exercises from the dataset.
    """
    plan_prediction = predict_plan_type(profile)

    goal = context.get('goal', 'maintenance')
    fitness_level = profile.get('fitness_level', 'beginner')
    days_per_week = context.get('days_per_week', 3)
    duration_weeks = context.get('duration_weeks', 4)

    weekly_exercises = select_exercises(goal, fitness_level, days_per_week)

    return {
        'plan_type': plan_prediction['plan_type'],
        'goal': goal,
        'fitness_level': fitness_level,
        'days_per_week': days_per_week,
        'duration_weeks': duration_weeks,
        'weekly_plan': weekly_exercises
    }