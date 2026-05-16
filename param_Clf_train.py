# src/train.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

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
    "model.pkl"
)

# LOAD DATA
# =========================================================

df = pd.read_parquet(DATA_PATH)

print("✅ Data Loaded")

# =========================================================
# FEATURES / TARGET
# =========================================================

DROP_COLS = [
    "student_id",
    "final_score",
    "final_grade_band",
    "passed"
]

X = df.drop(columns=DROP_COLS)

y = df["passed"]

# =========================================================
# SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================================================
# MODEL PIPELINE
# =========================================================

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
# =========================================================

model.fit(X_train, y_train)

print("✅ Model Trained")

# =========================================================
# TEST ACCURACY
# =========================================================

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc:.4f}")

# SAVE MODEL
# =========================================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("✅ Model Saved")
print(MODEL_PATH)