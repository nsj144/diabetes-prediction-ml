# Diabetes Prediction using Machine Learning

## Project summary

This project implements a supervised binary classification workflow using the diabetes dataset provided with the project. Logistic Regression is used to estimate the `Outcome` class from eight input features. The workflow includes data inspection, preprocessing, a stratified train/test split, model fitting, and evaluation.

## 1. Dataset and preparation

The dataset contains **768 records and 9 columns**. There are eight input features and one target column (`Outcome`). The target distribution is:

| Outcome | Meaning | Records |
|---:|---|---:|
| 0 | No Diabetes | 500 |
| 1 | Diabetes | 268 |

No blank cells were found in the supplied CSV. Zero values were found in five measurement columns:

| Column | Zero values |
|---|---:|
| `Glucose` | 5 |
| `BloodPressure` | 35 |
| `SkinThickness` | 227 |
| `Insulin` | 374 |
| `BMI` | 11 |

For this analysis, zero values in those five columns are treated as unavailable measurements. The zero count in `Pregnancies` and the target value `Outcome = 0` are retained. This is a preprocessing assumption and should be kept in mind when interpreting the results.

The data is split into **614 training records (80%)** and **154 testing records (20%)** using stratification and `random_state=42`. The input features are the eight columns listed in the dataset; `Outcome` is the target. No categorical encoding is needed because the supplied input features are numeric.

Missing measurements are filled using the median, and numerical features are standardized. Both transformations are part of a Scikit-learn pipeline and are fitted using the training data only.

## 2. Model selection

**Model:** Logistic Regression (`sklearn.linear_model.LogisticRegression`).

The target has two classes, making this a binary classification problem. Logistic Regression is used as an interpretable baseline and produces an estimated probability for the positive class. The input features are standardized because their numerical ranges differ.

## 3. Training and evaluation

The model was fitted using the 614 training records. Performance was then measured on the 154 records reserved for testing.

| Metric | Result |
|---|---:|
| Training accuracy | **79.6%** |
| Test accuracy | **70.8%** |
| Test precision — Diabetes (class 1) | **60.0%** |
| Test recall — Diabetes (class 1) | **50.0%** |
| Test F1-score — Diabetes (class 1) | **54.5%** |

### Confusion matrix

Rows represent actual labels; columns represent predicted labels.

| | Predicted No Diabetes (0) | Predicted Diabetes (1) |
|---|---:|---:|
| Actual No Diabetes (0) | 82 | 18 |
| Actual Diabetes (1) | 27 | 27 |

The model correctly classified 82 records from class 0 and 27 records from class 1. It produced 18 false positives and 27 false negatives on the test set.

## 4. Performance analysis

Training accuracy (79.6%) is about **8.9 percentage points** higher than test accuracy (70.8%). The difference indicates that performance was better on the training records than on the held-out records. It is worth monitoring, but a single split does not establish that the model is overfitting.

For class 1, precision of 60.0% means that 60% of the records predicted as Diabetes were labelled Diabetes in this test set. Recall of 50.0% means that the model identified half of the actual Diabetes-class records. The F1-score of 54.5% summarizes the balance between precision and recall. The 27 false negatives show why test recall should be considered alongside overall accuracy.

## Conclusion

The Logistic Regression pipeline completes the requested end-to-end classification workflow and provides a reproducible baseline for the supplied dataset. The test results show room for improvement, particularly in identifying the positive class. Further work could include cross-validation, comparison with other classifiers, and decision-threshold analysis. Any changes should be evaluated on data that was not used to fit the model.

The dataset is limited and the model has not been clinically validated. Its predictions must not be used for diagnosis, treatment, or other medical decisions.

## Reproduction

From the project root, install the packages listed in `requirements.txt` and run:

```bash
python src/train_model.py
```

The script prints the data exploration and evaluation results, then saves the fitted pipeline to `models/diabetes_model.joblib` and the run metrics to `models/evaluation_metrics.json`. The split is repeatable because `random_state=42` is specified in the source code.
