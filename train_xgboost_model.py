 # src/train_xgboost.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier

from pipeline import preprocessor

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "students_messy.parquet"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
)

# =========================================================
# LOAD DATA
df = pd.read_parquet(DATA_PATH)

print("✅ Data Loaded")
print(df.shape)

# =========================================================
# DROP LEAKAGE COLUMNS
# =========================================================
DROP_COLS = [
    "student_id",
    "final_score",
    "final_grade_band",
    "passed"
]

X = df.drop(columns=DROP_COLS)

# Binary classification target
y = df["passed"]

# SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("✅ Train/Test Split Complete")

# =========================================================
# XGBOOST PIPELINE

model = Pipeline([
    ("pre", preprocessor),

    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",

        n_estimators=312,
        max_depth=3,
        learning_rate=0.014,
        subsample=0.95,
        colsample_bytree=0.64,
        reg_lambda=2.95,
        reg_alpha=0.78,

        random_state=42
    ))
])
# =========================================================
# TRAIN

model.fit(X_train, y_train)

print("✅ XGBoost Model Trained")
# =======================================================
# PREDICTIONS

pred = model.predict(X_test)

proba = model.predict_proba(X_test)[:, 1]

# =========================================================
# METRICS
# =========================================================

acc = accuracy_score(y_test, pred)

f1 = f1_score(y_test, pred)

auc = roc_auc_score(y_test, proba)

print("\n==============================")
print("XGBOOST RESULTS")
print("==============================")

print(f"Accuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nClassification Report\n")

print(classification_report(y_test, pred))

# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\n✅ Model Saved Successfully")
print(MODEL_PATH)