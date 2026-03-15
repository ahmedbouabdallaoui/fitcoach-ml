import pandas as pd
import numpy as np
from xgboost import XGBClassifier
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
df = pd.read_csv(os.path.join(DATA_DIR, 'final_dataset_BFP.csv'))

print("Dataset shape:", df.shape)

# --- Merge plans 6 and 7 into one class ---
# Both represent high BMI users — boundary between them is too blurry to learn reliably
df['Exercise Recommendation Plan'] = df['Exercise Recommendation Plan'].replace(7, 6)

print("\nPlan distribution after merge:")
print(df['Exercise Recommendation Plan'].value_counts())

# Plan meanings after merge:
# 1 → severe thinness
# 2 → moderate thinness
# 3 → mild thinness
# 4 → normal
# 5 → overweight
# 6 → obese / severe obese (merged)

# --- Feature Engineering ---
df['Gender_Encoded'] = (df['Gender'] == 'Male').astype(int)
df['BMI_X_BFP'] = df['BMI'] * df['Body Fat Percentage']
df['Weight_X_BFP'] = df['Weight'] * df['Body Fat Percentage']
df['BMI_Squared'] = df['BMI'] ** 2
df['BFP_Squared'] = df['Body Fat Percentage'] ** 2
df['BMI_X_BFP_Squared'] = df['BMI_Squared'] * df['BFP_Squared']

df['BMI_Category_Score'] = pd.cut(
    df['BMI'],
    bins=[0, 18.5, 24.9, 29.9, 34.9, 100],
    labels=[1, 2, 3, 4, 5]
).astype(float)

df['BFP_Category_Score'] = pd.cut(
    df['Body Fat Percentage'],
    bins=[0, 6, 14, 18, 25, 100],
    labels=[1, 2, 3, 4, 5]
).astype(float)

features = [
    'Weight',
    'Height',
    'BMI',
    'Body Fat Percentage',
    'Age',
    'Gender_Encoded',
    'BMI_X_BFP',
    'Weight_X_BFP',
    'BMI_Squared',
    'BFP_Squared',
    'BMI_X_BFP_Squared',
    'BMI_Category_Score',
    'BFP_Category_Score',
]

X = df[features]
y = df['Exercise Recommendation Plan']

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
print("\nTuning XGBoost...")

param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    ),
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

plan_names = {
    1: 'severe_thinness',
    2: 'moderate_thinness',
    3: 'mild_thinness',
    4: 'normal',
    5: 'overweight',
    6: 'obese'
}

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=[plan_names[c] for c in le.classes_]
))

# --- Feature Importance ---
importances = pd.Series(
    best_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nTop features:")
print(importances)

# --- Save ---
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODELS_DIR, 'training_plan_model.pkl'))
joblib.dump(features, os.path.join(MODELS_DIR, 'training_plan_features.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'training_plan_label_encoder.pkl'))
joblib.dump(plan_names, os.path.join(MODELS_DIR, 'training_plan_names.pkl'))

print("\nTraining plan model saved!")