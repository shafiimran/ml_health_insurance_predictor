import streamlit as st
import pandas as pd
import joblib

st.header("Health Insurance Premium Predictor")

@st.cache_resource
def load_artifacts():
    model_young   = joblib.load("artifacts/model_young.joblib")
    scaler_young  = joblib.load("artifacts/scaler_young.joblib")
    model_rest    = joblib.load("artifacts/model_rest.joblib")
    scaler_rest   = joblib.load("artifacts/scaler_rest.joblib")
    feature_cols_young = joblib.load("artifacts/feature_columns.joblib")
    feature_cols_rest  = joblib.load("artifacts/feature_columns.joblib")
    return model_young, scaler_young, model_rest, scaler_rest, feature_cols_young, feature_cols_rest

model_young, scaler_young, model_rest, scaler_rest, feature_cols_young, feature_cols_rest = load_artifacts()

RISK_SCORE_MAP = {
    "No Disease": 0,
    "Diabetes": 6,
    "High Blood Pressure": 6,
    "Thyroid": 5,
    "Heart Disease": 8,
    "Diabetes & High Blood Pressure": 12,
    "Diabetes & Thyroid": 11,
    "Diabetes & Heart Disease": 14,
    "High Blood Pressure & Thyroid": 11,
    "High Blood Pressure & Heart Disease": 14,
    "Thyroid & Heart Disease": 13,
    "Diabetes & High Blood Pressure & Thyroid": 17,
    "Diabetes & High Blood Pressure & Heart Disease": 20,
    "Diabetes & Thyroid & Heart Disease": 19,
    "High Blood Pressure & Thyroid & Heart Disease": 19,
    "Diabetes & High Blood Pressure & Thyroid & Heart Disease": 25,
}

def preprocess(inputs, scaler_bundle, feature_cols):
    insurance_map = {"Bronze": 1, "Silver": 2, "Gold": 3}
    scaler       = scaler_bundle["scaler"]
    cols_to_scale = scaler_bundle["columns"]

    row = {
        "age":                   inputs["age"],
        "number_of_dependants":  inputs["number_of_dependants"],
        "income_lakhs":          inputs["income_lakhs"],
        "genetical_risk":        inputs["genetical_risk"],
        "insurance_plan":        insurance_map[inputs["insurance_plan"]],
        "normalized_risk_score": RISK_SCORE_MAP.get(inputs["medical_history"], 0) / 25,
        "gender":                inputs["gender"],
        "region":                inputs["region"],
        "marital_status":        inputs["marital_status"],
        "bmi_category":          inputs["bmi_category"],
        "smoking_status":        inputs["smoking_status"],
        "employment_status":     inputs["employment_status"],
    }
    df = pd.DataFrame([row])

    nominal_cols = ["gender", "region", "marital_status",
                    "bmi_category", "smoking_status", "employment_status"]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    df['income_level'] = None
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop(columns=['income_level'], inplace=True)
    return df


# ── Inputs ──
st.subheader("Personal Information")
col1, col2, col3 = st.columns(3)
age                  = col1.number_input("Age", min_value=18, max_value=100, value=25)
gender               = col2.selectbox("Gender", ["Male", "Female"])
marital_status       = col3.selectbox("Marital Status", ["Unmarried", "Married"])

col4, col5 = st.columns(2)
number_of_dependants = col4.number_input("Number of Dependants", min_value=0, max_value=10, value=0)
region               = col5.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

st.subheader("Health Profile")
col6, col7 = st.columns(2)
bmi_category   = col6.selectbox("BMI Category", ["Normal", "Overweight", "Underweight", "Obesity"])
smoking_status = col7.selectbox("Smoking Status", ["No Smoking", "Occasional", "Regular"])

col8, col9 = st.columns(2)
medical_history = col8.selectbox("Medical History", list(RISK_SCORE_MAP.keys()))
genetical_risk  = col9.number_input("Genetical Risk (0–5)", min_value=0, max_value=5, value=0)

st.subheader("Financial & Employment")
col10, col11, col12 = st.columns(3)
employment_status = col10.selectbox("Employment Status", ["Salaried", "Self-Employed", "Freelancer"])
income_level      = col11.selectbox("Income Level", ["<10L", "10L - 25L", "25L - 40L", "> 40L"])
income_lakhs      = col12.number_input("Income (Lakhs)", min_value=0, max_value=200, value=10)

insurance_plan = st.selectbox("Insurance Plan", ["Bronze", "Silver", "Gold"])

# ── Predict ──
if st.button("Predict Premium"):
    inputs = {
        "age": age, "gender": gender, "region": region,
        "marital_status": marital_status,
        "number_of_dependants": number_of_dependants,
        "bmi_category": bmi_category, "smoking_status": smoking_status,
        "employment_status": employment_status, "income_level": income_level,
        "income_lakhs": income_lakhs, "medical_history": medical_history,
        "insurance_plan": insurance_plan, "genetical_risk": genetical_risk,
    }
    if age < 25:
        model, scaler_bundle, feature_cols = model_young, scaler_young, feature_cols_young
    else:
        model, scaler_bundle, feature_cols = model_rest, scaler_rest, feature_cols_rest

    X = preprocess(inputs, scaler_bundle, feature_cols)
    prediction = model.predict(X)[0]
    st.success(f"Predicted Premium: {prediction:,.0f}")
