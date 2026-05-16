# src/api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import joblib
import os

# =========================
# APP
# =========================

app = FastAPI(title="Student Performance API", version="0.1")

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "model.pkl"
)

model = joblib.load(MODEL_PATH)

# =========================
# INPUT SCHEMA
# =========================

class StudentInput(BaseModel):

    gender: str
    school_type: str
    parent_edu: str

    prior_gpa: float
    attendance_pct: float
    quiz_avg: float
    assign_avg: float
    midterm: float

    study_hours_wk: float
    on_time_submit_pct: float
    lms_logins_wk: float
    forum_posts: float
    commute_min: float

# =========================
# ROOT
# =========================

@app.get("/")
def home():

    return {
        "message": "Student Performance API Running"
    }

# =========================
# PREDICT
# =========================

@app.post("/predict")
def predict(data: StudentInput):

    try:

        df = pd.DataFrame([data.dict()])

        probability = model.predict_proba(df)[0][1]

        prediction = int(probability >= 0.5)

        return {
            "prediction": prediction,
            "pass_probability": round(float(probability), 4)
        }

    except Exception as e:

        return {
            "error": str(e)
        }
    return {
         "risk_probability": 0.72,
        "at_risk": True,
        "prediction": int(prediction),
        "pass_probability": round(float(probability), 4)
    }
    