from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and feature columns
with open("C:\\Users\\Sanjeev Kumar\\Desktop\\heart_disease_predictor\\model.pkl", "rb") as f:
    model = pickle.load(f)
with open("C:\\Users\\Sanjeev Kumar\\Desktop\\heart_disease_predictor\\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("C:\\Users\\Sanjeev Kumar\\Desktop\\heart_disease_predictor\\feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get form values
            age = int(request.form["age"])
            avg_hr = int(request.form["avg_hr"])
            steps = int(request.form["steps"])
            daily_calories_burnt = float(request.form["daily_calories_burnt"])
            stress_level = int(request.form["stress_level"])
            gender = request.form["gender"]
            smoking_s = int(request.form["smoking_s"])
            family_he = int(request.form["family_he"])

            # Build dataframe
            input_dict = {
                "age": age,
                "avg_heart_rate": avg_hr,
                "steps_per_day": steps,
                "daily_calories_burnt": daily_calories_burnt,
                "stress_level": stress_level,
                "gender": gender,
                "smoking_status": smoking_s,
                "family_heart_disease": family_he,
            }

            input_df = pd.DataFrame([input_dict])

            # One-hot encode using same structure
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

            # Scale
            input_scaled = scaler.transform(input_encoded)

            # Predict
            result = model.predict(input_scaled)[0]
            prediction = "🚨 High Risk" if result == 1 else "✅ Low Risk"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
