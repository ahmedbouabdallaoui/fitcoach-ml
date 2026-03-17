import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'training', 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- Load Dataset ---
df = pd.read_csv(os.path.join(DATA_DIR, 'sports_multimodal_data.csv'))

print("Dataset shape:", df.shape)
print("\nInjury distribution:")
print(df['injury_risk'].value_counts())

# --- Feature Engineering ---
# Combine physiological stress signals
df['Cardio_Stress'] = df['heart_rate'] * df['workload_intensity']
df['BP_Pulse'] = df['bp_systolic'] - df['bp_diastolic']
df['Recovery_Load_Ratio'] = df['rest_period'] / (df['workload_intensity'].abs() + 1e-6)
df['Fatigue_X_Load'] = df['fatigue_index'] * df['workload_intensity'].abs()
df['Impact_X_Reps'] = df['impact_force'] * df['repetition_count']
df['Injury_X_Load'] = df['previous_injury_history'] * df['workload_intensity'].abs()
df['EMG_X_Duration'] = df['emg_amplitude'] * df['training_duration']

features = [
    # Core physiological
    'heart_rate',
    'emg_amplitude',
    'fatigue_index',
    'workload_intensity',
    'rest_period',
    'previous_injury_history',
    'training_duration',
    'repetition_count',
    'ground_reaction_force',
    'impact_force',
    'gait_symmetry',
    'range_of_motion',
    'spo2',
    'respiratory_rate',
    'bp_systolic',
    'bp_diastolic',
    # Engineered
    'Cardio_Stress',
    'BP_Pulse',
    'Recovery_Load_Ratio',
    'Fatigue_X_Load',
    'Impact_X_Reps',
    'Injury_X_Load',
    'EMG_X_Duration',
]

X = df[features]
y = df['injury_risk']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- SMOTE ---
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nTraining set after SMOTE: {len(X_train_balanced)} samples")
print(f"Balanced distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# --- Hyperparameter Tuning ---
print("\nTuning XGBoost...")

param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1]  # SMOTE already balanced, no need for extra weighting
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_balanced, y_train_balanced)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.2%}")

# --- Evaluate ---
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Injury', 'Injury Risk']))

# --- Feature Importance ---
importances = pd.Series(
    best_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nTop 10 features:")
print(importances.head(10))

# --- Save ---
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODELS_DIR, 'injury_prediction_model.pkl'))
joblib.dump(features, os.path.join(MODELS_DIR, 'injury_prediction_features.pkl'))

print("\nInjury prediction model saved!")