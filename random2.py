import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load your dataset
data = pd.read_csv('dataset.csv')  # Replace 'data.csv' with your actual dataset file path

# Define the features and target variable
X = data[['age', 'avg_heart_rate', 'steps_per_day', 'daily_calories_burnt', 'stress_level', 'gender', 'smoking_s', 'family_he']]
y = data['target']  # Assuming 'target' is the column you want to predict

# One-hot encode categorical features (gender, smoking_s, family_he)
X_encoded = pd.get_dummies(X, drop_first=True)  # drop_first=True to avoid dummy variable trap

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the scaler and apply it to the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the RandomForestClassifier and fit the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model, scaler, and feature columns
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('feature_columns.pkl', 'wb') as feature_file:
    pickle.dump(X_encoded.columns.tolist(), feature_file)

print("Model, scaler, and feature columns saved successfully.")
