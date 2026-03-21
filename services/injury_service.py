import joblib
import pandas as pd
import numpy as np
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- Load model once at startup ---
injury_model = joblib.load(os.path.join(MODELS_DIR, 'injury_prediction_model.pkl'))
injury_features = joblib.load(os.path.join(MODELS_DIR, 'injury_prediction_features.pkl'))


def predict_injury_risk(profile: dict, context: dict) -> dict:
    """
    Predicts injury risk based on physical profile and training context.
    Estimates physiological values from available profile data.
    """

    weight = profile.get('weight_kg', 70)
    height_cm = profile.get('height_cm', 170)
    age = profile.get('age', 25)
    weekly_hours = context.get('weekly_training_hours', 5)
    has_previous_injuries = 1 if context.get('has_previous_injuries', False) else 0
    fitness_level = profile.get('fitness_level', 'beginner')

    height_m = height_cm / 100
    bmi = weight / (height_m ** 2)

    # --- Estimate training intensity from weekly hours ---
    if weekly_hours <= 3:
        training_intensity = 0.3
    elif weekly_hours <= 6:
        training_intensity = 0.5
    elif weekly_hours <= 10:
        training_intensity = 0.7
    else:
        training_intensity = 0.9

    # --- Estimate rest period from fitness level ---
    fitness_map = {'beginner': 2, 'intermediate': 4, 'advanced': 6}
    rest_period = fitness_map.get(fitness_level, 3)

    # --- Estimate physiological values from profile ---
    # These are medically reasonable estimates based on fitness level
    fitness_hr_map = {'beginner': 75, 'intermediate': 68, 'advanced': 60}
    resting_hr = fitness_hr_map.get(fitness_level, 70)
    heart_rate = resting_hr + (training_intensity * 80)

    training_duration = weekly_hours * 60 / 7  # avg daily minutes
    repetition_count = int(10 + training_intensity * 20)

    # Physiological estimates
    emg_amplitude = 0.3 + training_intensity * 0.4
    fatigue_index = 30 + training_intensity * 40 + (has_previous_injuries * 10)
    workload_intensity = training_intensity * 8
    ground_reaction_force = weight * 9.8 * (1 + training_intensity * 0.3)
    impact_force = ground_reaction_force * (1.2 + training_intensity * 0.5)
    gait_symmetry = 0.95 - (training_intensity * 0.1) - (has_previous_injuries * 0.05)
    range_of_motion = 90 + (30 * (1 - training_intensity))
    spo2 = 98 - training_intensity
    respiratory_rate = 15 + training_intensity * 8
    bp_systolic = 120 + training_intensity * 20
    bp_diastolic = 80 + training_intensity * 10
    acc_rms = 0.9 + training_intensity * 0.3

    # --- Engineered features ---
    cardio_stress = heart_rate * workload_intensity
    bp_pulse = bp_systolic - bp_diastolic
    recovery_load_ratio = rest_period / (workload_intensity + 1e-6)
    fatigue_x_load = fatigue_index * workload_intensity
    impact_x_reps = impact_force * repetition_count
    injury_x_load = has_previous_injuries * workload_intensity
    emg_x_duration = emg_amplitude * training_duration

    input_data = pd.DataFrame([{
        'heart_rate': heart_rate,
        'emg_amplitude': emg_amplitude,
        'fatigue_index': fatigue_index,
        'workload_intensity': workload_intensity,
        'rest_period': rest_period,
        'previous_injury_history': has_previous_injuries,
        'training_duration': training_duration,
        'repetition_count': repetition_count,
        'ground_reaction_force': ground_reaction_force,
        'impact_force': impact_force,
        'gait_symmetry': gait_symmetry,
        'range_of_motion': range_of_motion,
        'spo2': spo2,
        'respiratory_rate': respiratory_rate,
        'bp_systolic': bp_systolic,
        'bp_diastolic': bp_diastolic,
        'Cardio_Stress': cardio_stress,
        'BP_Pulse': bp_pulse,
        'Recovery_Load_Ratio': recovery_load_ratio,
        'Fatigue_X_Load': fatigue_x_load,
        'Impact_X_Reps': impact_x_reps,
        'Injury_X_Load': injury_x_load,
        'EMG_X_Duration': emg_x_duration
    }])

    # --- Predict ---
    prediction = injury_model.predict(input_data[injury_features])[0]
    probability = injury_model.predict_proba(input_data[injury_features])[0]

    # --- Convert to 0-10 risk score ---
    risk_score = round(probability[1] * 10, 1)

    # --- Risk level ---
    if risk_score < 3:
        risk_level = 'Low'
    elif risk_score < 6:
        risk_level = 'Medium'
    elif risk_score < 8:
        risk_level = 'High'
    else:
        risk_level = 'Critical'

    # --- Risk factors ---
    risk_factors = []
    if has_previous_injuries:
        risk_factors.append("Previous injury history increases re-injury risk")
    if training_intensity >= 0.7:
        risk_factors.append("High training intensity detected")
    if weekly_hours > 10:
        risk_factors.append("Training volume exceeds recommended weekly hours")
    if rest_period <= 2:
        risk_factors.append("Insufficient recovery time between sessions")
    if age > 35:
        risk_factors.append("Age factor — recovery takes longer after 35")
    if bmi > 29.9:
        risk_factors.append("Elevated BMI increases joint stress")

    if not risk_factors:
        risk_factors.append("No significant risk factors detected")

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'high_risk': risk_score >= 7,
        'prediction': int(prediction),
        'risk_factors': risk_factors
    }