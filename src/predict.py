"""Run an interactive prediction with the saved diabetes model."""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_model.joblib"


def main() -> None:
    if not MODEL_PATH.is_file():
        print("Model file not found. Run 'python src/train_model.py' first.")
        return

    saved_model = joblib.load(MODEL_PATH)
    model = saved_model["model"]
    feature_columns = saved_model["feature_columns"]

    print("Enter one set of feature values.")
    print("Use 0 if a Glucose, BloodPressure, SkinThickness, Insulin, or BMI measurement is unavailable.\n")

    values = {}
    for column in feature_columns:
        while True:
            entered_value = input(f"{column}: ").strip()
            try:
                values[column] = float(entered_value)
                break
            except ValueError:
                print("Please enter a numeric value.")

    input_record = pd.DataFrame([values], columns=feature_columns)
    prediction = int(model.predict(input_record)[0])
    probability = float(model.predict_proba(input_record)[0][1])

    print("\nPREDICTION")
    print(f"Predicted class: {prediction} ({'Diabetes' if prediction == 1 else 'No Diabetes'})")
    print(f"Estimated probability for class 1: {probability:.1%}")
    print("For educational use only; this output is not a medical diagnosis.")


if __name__ == "__main__":
    main()
