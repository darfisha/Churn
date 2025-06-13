import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time
import os

# ✅ THIS MUST BE FIRST
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

# ---------------------------- #
# 🚀 Data Preparation & Model
# ---------------------------- #

@st.cache_resource
def train_model():
    if not os.path.exists("Churn.csv"):
        st.error("❗ Dataset file 'Churn.csv' not found. Please upload it.")
        st.stop()

    df = pd.read_csv("Churn.csv")
    df = df.drop(columns=["customer_id"])
    df["country"] = df["country"].map({"France": 0, "Spain": 1, "Germany": 2})
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    df["credit_card"] = df["credit_card"].map({"Yes": 1, "No": 0})
    df["active_member"] = df["active_member"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    with open("churn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return model, scaler, report

model, scaler, report = train_model()

st.title("📊 Customer Churn Prediction")

# ---------------------------- #
# 📋 User Inputs
# ---------------------------- #

country = st.selectbox("Country", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
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

input_data = np.array([[
    country_map[country],
    gender_map[gender],
    age,
    tenure,
    balance,
    products_number,
    credit_card_map[credit_card],
    active_member_map[active_member],
    estimated_salary
]])

input_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    with st.spinner("Predicting..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ This customer is likely to churn! (Probability: {probability:.2f}%)")
        st.snow()
    else:
        st.success(f"✅ This customer is unlikely to churn. (Probability: {probability:.2f}%)")
        st.balloons()

with st.expander("Show Model Evaluation Metrics"):
    st.write("Classification Report (on test set):")
    st.json(report)
