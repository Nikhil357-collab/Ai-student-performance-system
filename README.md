# Ai-student-performance-system
An end-to-end Machine Learning + Full Stack AI application that predicts whether a student is likely to pass based on academic and behavioral features.
# 🎓 AI Student Performance Prediction System

An end-to-end Machine Learning + Full Stack AI application that predicts whether a student is likely to pass based on academic and behavioral features.
---

# 🚀 Project Overview

This project evolved from a simple ML notebook into a production-style AI system with:

- Machine Learning Pipeline
- XGBoost Model
- FastAPI Backend
- Next.js Frontend
- SHAP Explainability
- Interactive Dashboard
- Probability Visualizations
- SQLite Database Integration

# 🧠 Problem Statement

Educational institutions often identify weak-performing students too late.

This system predicts student performance early using:
- attendance
- GPA
- quiz scores
- assignment scores
- LMS activity
- study behavior

allowing early intervention and support.

---

# ⚙️ Tech Stack

## Machine Learning
- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Optuna

## Backend
- FastAPI
- Uvicorn
- Joblib
- Pydantic

## Frontend
- Next.js
- TypeScript
- Tailwind CSS
- Framer Motion
- Recharts
- Axios

## Database
- SQLite
- Prisma ORM

---

# 📊 Features

## ML Features
- Student pass/fail prediction
- Probability scoring
- Feature engineering
- Hyperparameter tuning
- Model evaluation
- Fairness analysis

## Dashboard Features
- Interactive prediction form
- Probability gauge
- Radar performance chart
- Glassmorphism UI
- SHAP explanation cards
- Responsive design

## Backend Features
- REST API
- Prediction endpoint
- Validation using Pydantic
- CORS integration

---

# 🧪 Model Performance

| Model | F1 Score |
|---|---|
| Logistic Regression | 0.677 |
| Random Forest | 0.801 |
| XGBoost | 0.980 |

---

# 📁 Project Structure

```bash
STUDENT_PERFORMANCE/
│
├── src/
│   ├── api.py
│   ├── train_xgboost.py
│   ├── evaluate.py
│   └── pipeline.py
│
├── models/
│   └── model.pkl
│
├── data/
│
├── frontend/
│
└── notebooks/
