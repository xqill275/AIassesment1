from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# -------------------------
# Load Model + Scaler
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "bestModel.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Load a sample row to get correct column order after encoding
sample_data = pd.read_csv(os.path.join(BASE_DIR, "data/heart-disease-dataset.csv"))
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Create column template using original encoding
template_df = pd.get_dummies(sample_data.drop("HeartDisease", axis=1),
                             columns=categorical_cols, drop_first=True)
template_cols = template_df.columns


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user input
        user_input = {
            "Age": float(request.form["Age"]),
            "RestingBP": float(request.form["RestingBP"]),
            "Cholesterol": float(request.form["Cholesterol"]),
            "FastingBS": float(request.form["FastingBS"]),
            "MaxHR": float(request.form["MaxHR"]),
            "Oldpeak": float(request.form["Oldpeak"]),
            "Sex": request.form["Sex"],
            "ChestPainType": request.form["ChestPainType"],
            "RestingECG": request.form["RestingECG"],
            "ExerciseAngina": request.form["ExerciseAngina"],
            "ST_Slope": request.form["ST_Slope"]
        }

        # Convert to DataFrame
        df = pd.DataFrame([user_input])

        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Add any missing columns
        for col in template_cols:
            if col not in df_encoded:
                df_encoded[col] = 0

        # Reorder columns
        df_encoded = df_encoded[template_cols]

        # Scale numeric data
        df_scaled = scaler.transform(df_encoded)

        # Predict
        prediction = model.predict(df_scaled)[0]

        if prediction == 1:
            result = "You may be at HIGH risk of Heart Disease."
        else:
            result = "You appear to be at LOW risk of Heart Disease."

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
