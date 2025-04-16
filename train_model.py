import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# 📁 Manually define where to save model files
output_dir = ("C:\\Users\\Sanjeev Kumar\\Desktop\\heart_disease_predictor")
os.makedirs(output_dir, exist_ok=True)

try:
    print("📥 Loading dataset...")
    data = pd.read_csv("C:\\Users\\Sanjeev Kumar\\Desktop\\heart_disease_predictor\\dataset.csv")  # Replace if needed
    print("✅ Dataset loaded.")
    print("📋 Columns:", data.columns.tolist())

    # ✅ Check required columns
    required = ['age', 'avg_heart_rate', 'steps_per_day', 'daily_calories_burnt',
                'stress_level', 'gender', 'smoking_status', 'family_heart_disease', 'heart_attack_risk']
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("⚙️ Preparing training data...")
    X = data[required[:-1]]
    y = data['heart_attack_risk']

    # One-hot encoding
    print("🔠 One-hot encoding...")
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()

    # Train/test split
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Scaling
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("🌲 Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)

    # 💾 Save files manually to Desktop or any specified folder
    print(f"💾 Saving files to: {output_dir}")
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_columns, f)

    print("✅ All files saved successfully!")

except Exception as e:
    print("❌ ERROR:", e)
