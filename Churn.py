import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time

# Set page configuration early
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

# --------- Custom CSS for Background & Styling ---------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1605902711622-cfb43c4437d1");
    background-size: cover;
    background-position: center;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

h1 {
    color: #ffffff;
    text-align: center;
    font-size: 3em;
    text-shadow: 2px 2px 4px #000000;
}

.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
    font-size: 1em;
}

.stButton button:hover {
    background-color: #45a049;
    transition: 0.3s;
}

.stNumberInput input, .stSelectbox div {
    border-radius: 10px !important;
}

div[data-testid="stExpander"] > div > div {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 10px;
}
footer {
    visibility: hidden;
}

footer:after {
    content:'Made by Darfisha';
    visibility: visible;
    display: block;
    text-align: center;
    color: white;
    padding: 10px;
    font-weight: bold;
    font-size: 14px;
    text-shadow: 1px 1px 2px #000000;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_resource
def train_model():
    df = pd.read_csv("Bank Customer Churn Prediction.csv")
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

# ----------------- App Layout -----------------
st.title("📊 Customer Churn Prediction")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- USER INPUT -----------------
with st.form("user_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox("Country", ["France", "Spain", "Germany"])
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0)
        credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
        products_number = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        active_member = st.selectbox("Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

    submitted = st.form_submit_button("Predict Churn")

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

if submitted:
    with st.spinner("Predicting..."):
        time.sleep(1)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ Likely to Churn! (Probability: {probability:.2f}%)")
    else:
        st.success(f"✅ Unlikely to Churn (Probability: {probability:.2f}%)")

with st.expander("🔎 Show Model Metrics"):
    st.json(report)
