import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'training', 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- Load Dataset ---
df = pd.read_csv(os.path.join(DATA_DIR, 'gym_members_exercise_tracking.csv'))

# --- Label Assignment ---
def assign_realistic_goal(row):
    bmi = row['BMI']
    experience = row['Experience_Level']
    fat = row['Fat_Percentage']
    freq = row['Workout_Frequency (days/week)']
    duration = row['Session_Duration (hours)']

    if bmi < 18.5 or (fat < 10 and experience == 1):
        return 'muscle_gain'
    elif bmi > 29.9 or fat > 30:
        return 'weight_loss'
    elif experience == 1 or freq < 3 or duration < 0.5:
        return 'endurance'
    else:
        return 'any'

df['Realistic_Goal'] = df.apply(assign_realistic_goal, axis=1)

print("Label distribution before SMOTE:")
print(df['Realistic_Goal'].value_counts())

# --- Feature Engineering ---
df['Gender_Encoded'] = (df['Gender'] == 'Male').astype(int)
df['Calories_Per_Hour'] = df['Calories_Burned'] / df['Session_Duration (hours)']
df['BPM_Range'] = df['Max_BPM'] - df['Resting_BPM']
df['BMI_X_Experience'] = df['BMI'] * df['Experience_Level']
df['Fat_X_Frequency'] = df['Fat_Percentage'] * df['Workout_Frequency (days/week)']
df['Intensity_Score'] = (df['Avg_BPM'] / df['Max_BPM']) * df['Session_Duration (hours)']
df['Volume_Score'] = df['Session_Duration (hours)'] * df['Workout_Frequency (days/week)']

features = [
    'Age',
    'Weight (kg)',
    'Height (m)',
    'BMI',
    'Experience_Level',
    'Fat_Percentage',
    'Workout_Frequency (days/week)',
    'Session_Duration (hours)',
    'Calories_Burned',
    'Avg_BPM',
    'Resting_BPM',
    'Max_BPM',
    'Water_Intake (liters)',
    'Gender_Encoded',
    'Calories_Per_Hour',
    'BPM_Range',
    'BMI_X_Experience',
    'Fat_X_Frequency',
    'Intensity_Score',
    'Volume_Score',
]

X = df[features]
y = df['Realistic_Goal']

# --- Encode Labels ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- SMOTE ---
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nTraining set after SMOTE: {len(X_train_balanced)} samples")

# --- Hyperparameter Tuning ---
print("\nTuning RandomForest...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_balanced, y_train_balanced)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.2%}")

# --- Evaluate ---
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Feature Importance ---
importances = pd.Series(
    best_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nTop 10 most important features:")
print(importances.head(10))

# --- Save ---
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODELS_DIR, 'goal_validator_model.pkl'))
joblib.dump(features, os.path.join(MODELS_DIR, 'goal_validator_features.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'goal_validator_label_encoder.pkl'))

print("\nGoal validator model saved!")