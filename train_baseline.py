# src/train_baselines.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pipeline import preprocessor  # your ColumnTransformer



# Load data
df = pd.read_parquet("data\\students_messy.parquet")

# Define features and target
X = df.drop(columns=["final_score", "final_grade_band", "passed"])
y = df["passed"]

# Models
logit = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced"))
])

rf = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])


# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate models
for name, model in [("Logistic Regression", logit), ("Random Forest", rf)]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
    
    print(f"\n{name}")
    print(f"F1 Mean: {scores.mean():.4f}")
    print(f"F1 Std:  {scores.std():6f}")