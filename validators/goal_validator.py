import joblib
import pandas as pd
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- Load models once at startup ---
goal_validator_model = joblib.load(os.path.join(MODELS_DIR, 'goal_validator_model.pkl'))
goal_validator_features = joblib.load(os.path.join(MODELS_DIR, 'goal_validator_features.pkl'))
goal_validator_le = joblib.load(os.path.join(MODELS_DIR, 'goal_validator_label_encoder.pkl'))


def validate_goal(profile: dict, user_goal: str) -> dict:
    """
    Validates if the user's goal is realistic based on their profile.
    Returns realistic_goal, goal_valid flag and optional warning message.
    """

    # --- Build BMI ---
    weight = profile.get('weight_kg', 70)
    height_m = profile.get('height_cm', 170) / 100
    bmi = weight / (height_m ** 2)

    # --- Experience level mapping ---
    fitness_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
    experience = fitness_map.get(profile.get('fitness_level', 'beginner'), 1)

    # --- Feature vector ---
    fat = profile.get('body_fat_percentage') or (bmi * 0.8)
    freq = 3
    duration = 1.0

    gender_encoded = 1 if profile.get('gender', 'Male').lower() == 'male' else 0

    calories_per_hour = 400
    bpm_range = 80
    bmi_x_experience = bmi * experience
    fat_x_frequency = fat * freq
    intensity_score = 0.7 * duration
    volume_score = duration * freq

    input_data = pd.DataFrame([{
        'Age': profile.get('age', 25),
        'Weight (kg)': weight,
        'Height (m)': height_m,
        'BMI': bmi,
        'Experience_Level': experience,
        'Fat_Percentage': fat,
        'Workout_Frequency (days/week)': freq,
        'Session_Duration (hours)': duration,
        'Calories_Burned': 400,
        'Avg_BPM': 130,
        'Resting_BPM': 60,
        'Max_BPM': 180,
        'Water_Intake (liters)': 2.5,
        'Gender_Encoded': gender_encoded,
        'Calories_Per_Hour': calories_per_hour,
        'BPM_Range': bpm_range,
        'BMI_X_Experience': bmi_x_experience,
        'Fat_X_Frequency': fat_x_frequency,
        'Intensity_Score': intensity_score,
        'Volume_Score': volume_score,
    }])

    # --- Predict ---
    pred_encoded = goal_validator_model.predict(input_data[goal_validator_features])
    realistic_goal = goal_validator_le.inverse_transform(pred_encoded)[0]

    # --- Check if user goal matches ---
    goal_valid = (realistic_goal == 'any' or realistic_goal == user_goal)

    warning_message = None
    recommended_goal = user_goal

    if not goal_valid:
        recommended_goal = realistic_goal
        warning_message = (
            f"Based on your profile, '{realistic_goal}' might be more suitable "
            f"than '{user_goal}'. Would you like to proceed with your original goal anyway?"
        )

    return {
        'realistic_goal': realistic_goal,
        'goal_valid': goal_valid,
        'recommended_goal': recommended_goal,
        'warning_message': warning_message
    }