# 🛡️ Health Insurance Premium Predictor

A machine learning web app that predicts annual health insurance premiums based on an applicant's personal, health, and financial profile.

**Live Demo → [ml-health-insurance-predictstreamlit.app](https://ml-health-insurance-predictstreamlit.app)**

---

## Overview

This project trains two separate Linear Regression models — one for applicants aged below 25 and one for the rest — and deploys them through a Streamlit frontend. The app takes user inputs and returns an estimated annual premium in real time.

## Features

- Dual model system based on age group (under 25 / 25 and above)
- Handles ordinal encoding, one-hot encoding, and MinMax scaling at inference time
- Medical history risk scoring derived from domain knowledge
- Clean, minimal UI built entirely with Streamlit's built-in components

## Tech Stack

- **Python** — data processing and modeling
- **Pandas, NumPy** — data manipulation
- **Scikit-learn** — model training, preprocessing
- **Statsmodels** — VIF analysis for multicollinearity
- **Streamlit** — frontend and deployment
- **Joblib** — model serialization

## Project Structure

```
├── app.py                  # Streamlit frontend
├── ml_premium_prediction.ipynb              # Data cleaning, EDA, feature engineering, model training
├── requirements.txt
└── artifacts/
    ├── model_young.joblib
    ├── model_rest.joblib
    ├── scaler_young.joblib
    ├── scaler_rest.joblib
    ├── feature_columns.joblib
```

## Input Features

| Feature | Type |
|---|---|
| Age | Numerical |
| Gender | Categorical |
| Region | Categorical |
| Marital Status | Categorical |
| Number of Dependants | Numerical |
| BMI Category | Categorical |
| Smoking Status | Categorical |
| Medical History | Categorical |
| Genetical Risk | Numerical |
| Employment Status | Categorical |
| Income Level | Categorical |
| Income (Lakhs) | Numerical |
| Insurance Plan | Categorical |

## How to Run Locally

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py
```

## Model Performance

| Model | R² (Train) | R² (Test) |
|---|---|---|
| Linear Regression (Young) | 0.988 | 0.989 |
| XG Boost (Rest) | — | — |
