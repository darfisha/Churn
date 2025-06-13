import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

@st.cache_resource
def train_model():
    df = pd.read_csv("Bank Customer Churn Prediction.csv")

    # Drop unused column
    df = df.drop(columns=["customer_id"])

    # Encode categorical variables
    df["country"] = df["country"].map({"France": 0, "Spain": 1, "Germany": 2})
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

    X = df.drop(columns=["churn"])
    y = df["churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, scaler, X.columns.tolist(), report

model, scaler, feature_columns, report = train_model()

st.title("📊 Bank Customer Churn Prediction")

# ----------------- USER INPUT -----------------
country = st.selectbox("Country", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active_member = st.selectbox("Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

country_map = {"France": 0, "Spain": 1, "Germany": 2}
gender_map = {"Male": 0, "Female": 1}
credit_card_map = {"Yes": 1, "No": 0}
active_member_map = {"Yes": 1, "No": 0}

input_dict = {
    "credit_score": credit_score,
    "country": country_map[country],
    "gender": gender_map[gender],
    "age": age,
    "tenure": tenure,
    "balance": balance,
    "products_number": products_number,
    "credit_card": credit_card_map[credit_card],
    "active_member": active_member_map[active_member],
    "estimated_salary": estimated_salary,
}

input_data = np.array([[input_dict[col] for col in feature_columns]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    with st.spinner("Predicting..."):
        for i in range(50):
            time.sleep(0.01)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ Likely to Churn! (Probability: {probability:.2f}%)")
    else:
        st.success(f"✅ Unlikely to Churn (Probability: {probability:.2f}%)")

with st.expander("Show Model Metrics"):
    st.json(report)
