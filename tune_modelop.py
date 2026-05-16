# src/tune_optuna.py
import pandas as pd
import optuna
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from pipeline import preprocessor as pre

# Load data
df = pd.read_parquet("data/students_messy.parquet")

# Features & target (leakage removed)
X = df.drop(columns=["final_score","final_grade_band","passed"])
y = df["passed"]

# Train/validation split
Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🎯 Optuna objective
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist"
    }

    model = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(**params))
    ])

    model.fit(Xtr, ytr)
    preds = model.predict(Xva)

    return f1_score(yva, preds)

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print("Best Params:", best_params)


# Train final model
final_model = Pipeline([
    ("pre", pre),
    ("clf", XGBClassifier(**best_params))
])

# Calibration
calibrated_model = CalibratedClassifierCV(
    final_model,
    method="isotonic",   # better but needs more data
    cv=3
)

calibrated_model.fit(Xtr, ytr)

