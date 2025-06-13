import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import streamlit as st

# Load data
df = pd.read_csv("Churn.py.csv")
# Data preprocessing


# Drop unused column
df = df.drop(columns=["customer_id"])

# Encode categorical variables
df["country"] = df["country"].map({"France": 0, "Spain": 1, "Germany": 2})
df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
df["credit_card"] = df["credit_card"].map({"Yes": 1, "No": 0})
df["active_member"] = df["active_member"].map({"Yes": 1, "No": 0})

# Features and target
X = df.drop(columns=["churn"])
y = df["churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
# Streamlit deployment

st.title("Customer Churn Prediction")

# Input fields for user
country = st.selectbox("Country", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active_member = st.selectbox("Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Map inputs as in training
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

if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is unlikely to churn.")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
