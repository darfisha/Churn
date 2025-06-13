import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

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
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
