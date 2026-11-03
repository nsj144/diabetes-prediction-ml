"""Train and evaluate the diabetes classification model."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

TARGET_COLUMN = "Outcome"
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_data() -> pd.DataFrame:
    """Load the dataset and check that the required columns are available."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Expected data/diabetes.csv."
        )

    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Required columns are missing from the dataset: {missing_columns}")

    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"The target column '{TARGET_COLUMN}' contains missing values.")

    target_values = set(data[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"The target column must contain only 0 and 1; found {sorted(target_values)}."
        )

    return data


def build_model() -> Pipeline:
    """Create the preprocessing and classification pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def main() -> None:
    data = load_data()

    print("\nDATA EXPLORATION")
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print("\nData types:")
    print(data.dtypes)
    print("\nMissing values in the original CSV:")
    print(data.isna().sum())
    print("\nDescriptive statistics:")
    print(data.describe().round(2))
    print("\nTarget distribution:")
    print(data[TARGET_COLUMN].value_counts().sort_index())
    print("\nZero values in selected measurement columns:")
    print((data[ZERO_AS_MISSING] == 0).sum())

    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nDATA PREPARATION")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records:  {len(X_test)}")
    print(f"Input features: {FEATURE_COLUMNS}")
    print(f"Target column: {TARGET_COLUMN}")

    model = build_model()
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions, zero_division=0)
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])

    print("\nMODEL")
    print("Algorithm: Logistic Regression")
    print("Preprocessing: median imputation and standard scaling")

    print("\nMODEL EVALUATION")
    print(f"Training accuracy: {train_accuracy:.3f} ({train_accuracy:.1%})")
    print(f"Test accuracy:     {test_accuracy:.3f} ({test_accuracy:.1%})")
    print(f"Precision (class 1): {precision:.3f}")
    print(f"Recall (class 1):    {recall:.3f}")
    print(f"F1-score (class 1):  {f1:.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["No Diabetes (0)", "Diabetes (1)"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                 Predicted 0   Predicted 1")
    print(f"Actual 0             {matrix[0, 0]:>4}            {matrix[0, 1]:>4}")
    print(f"Actual 1             {matrix[1, 0]:>4}            {matrix[1, 1]:>4}")

    accuracy_gap = train_accuracy - test_accuracy
    print("\nTRAINING AND TEST COMPARISON")
    print(f"Training minus test accuracy: {accuracy_gap:.1%}")
    print(
        "The training score is higher than the test score. Review this difference "
        "alongside cross-validation results before drawing conclusions about fit."
        if accuracy_gap > 0
        else "The test score is at least as high as the training score on this split."
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS},
        MODEL_PATH,
    )

    metrics = {
        "training_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "precision_class_1": round(float(precision), 4),
        "recall_class_1": round(float(recall), 4),
        "f1_score_class_1": round(float(f1), 4),
        "confusion_matrix_labels_0_then_1": matrix.tolist(),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "random_state": 42,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print(f"\nSaved model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print("Training and evaluation completed.")


if __name__ == "__main__":
    main()
