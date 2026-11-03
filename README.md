# Diabetes Prediction with Logistic Regression

## Overview

This project builds a binary classification model using the supplied diabetes dataset. It uses eight patient-related measurements to estimate the dataset's `Outcome` label (`0` = No Diabetes, `1` = Diabetes). The implementation covers data inspection, preprocessing, model fitting, and evaluation on a held-out test set.

This repository is intended for learning and review. The model has not been clinically validated and must not be used for diagnosis, treatment, or medical decisions.

## Repository contents

```text
diabetes-ml-project/
├── data/
│   └── diabetes.csv
├── src/
│   ├── train_model.py
│   └── predict.py
├── .gitignore
├── requirements.txt
├── README.md
├── METHODOLOGY.md
└── MODEL_EVALUATION.md
```

The `models/` directory is created when training runs. It contains the fitted model pipeline and metrics from that run; these generated artifacts are excluded from version control.

## Dataset

The CSV contains 768 records and 9 columns. Eight columns are used as input features, and `Outcome` is the target.

| Column | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin fold thickness |
| `Insulin` | Serum insulin level |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes pedigree score |
| `Age` | Patient age |
| `Outcome` | Target: 0 = No Diabetes; 1 = Diabetes |

The target distribution is 500 records in class 0 and 268 in class 1. The CSV contains no blank cells. Zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as unavailable measurements for this analysis and imputed during preprocessing. Zero values in `Pregnancies` and `Outcome` are retained. This is an explicit modelling assumption, not a change to the source data.

## Approach

- Split the data into 80% training and 20% testing, using stratification and `random_state=42`.
- Use median imputation for the selected unavailable measurements.
- Standardize the numerical inputs.
- Fit a Logistic Regression classifier.
- Evaluate accuracy, precision, recall, F1-score, and the confusion matrix.

Imputation, scaling, and classification are combined in a Scikit-learn pipeline. The imputer and scaler are fitted on the training data only, then applied to the test data. Additional implementation details are in [METHODOLOGY.md](METHODOLOGY.md), and the measured results are in [MODEL_EVALUATION.md](MODEL_EVALUATION.md).

## Environment

- Python 3.10 or later
- pip

Install the dependencies from the project root:

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, apply the setting to the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

This setting does not change the machine-wide execution policy. Alternatively, in Windows Command Prompt, use `.venv\Scripts\activate.bat`.

**macOS/Linux:**

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train and evaluate

Run from the project root:

```bash
python src/train_model.py
```

The script prints the dataset summary, split sizes, classification metrics, and confusion matrix. It writes the following files to `models/`:

- `diabetes_model.joblib` — fitted preprocessing and classification pipeline.
- `evaluation_metrics.json` — metric values, confusion matrix, and split sizes.

## Generate a prediction

After training, run:

```bash
python src/predict.py
```

Enter a numeric value for each feature when prompted. For a measurement in the five columns listed in the Dataset section that is unavailable, enter `0`. The script displays the predicted class and the model-estimated probability for class 1.

## Reference performance

With the supplied CSV, the current preprocessing, and the fixed split, the model reports 79.6% training accuracy and 70.8% test accuracy. For class 1, test precision is 60.0%, recall is 50.0%, and F1-score is 54.5%. These metrics describe this specific hold-out split; they are not a guarantee of performance on other data.

## Limitations

This implementation is a baseline evaluated on a single train/test split. In particular, the test recall and false-negative count should be considered alongside accuracy. Cross-validation, comparison with other classifiers, and threshold analysis would provide a more complete evaluation. The dataset and model have not been validated for clinical use.
