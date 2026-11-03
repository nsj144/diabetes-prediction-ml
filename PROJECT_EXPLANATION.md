# Project Methodology

## Objective

The objective is to build a binary classifier using the supplied diabetes dataset. The model uses eight patient-related attributes to predict the `Outcome` label:

- `0` — No Diabetes
- `1` — Diabetes

This is a supervised learning task because the dataset includes an outcome label for each record.

## 1. Data preparation

The dataset is loaded into a Pandas DataFrame. The program reports the number of rows and columns, column names, data types, blank values, descriptive statistics, outcome counts, and zero counts in selected measurement columns.

The eight input features are `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`. The target variable is `Outcome`.

The CSV contains no blank cells, but zero values are present in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`. For this project, those zeros are treated as unavailable measurements. Zero values in `Pregnancies` and `Outcome` are kept unchanged.

The data is divided into 80% training records and 20% testing records. Stratification keeps the target-class proportions similar across the two sets. The fixed random state makes the split reproducible.

A Scikit-learn pipeline performs median imputation and standard scaling before Logistic Regression. The imputer and scaler are fitted on the training records only. The same fitted transformations are applied to the testing records.

There are no categorical input variables in the supplied dataset, so categorical encoding is not required.

## 2. Model selection

Logistic Regression is used as the baseline classifier. It is appropriate for an initial two-class classification exercise and provides both a class prediction and an estimated probability for the positive class. Standardization is applied because the numerical input features have different scales.

## 3. Training and evaluation

The pipeline is fitted using the training records. Predictions are generated for both training and testing records. The script reports:

- **Accuracy:** the proportion of predictions that are correct.
- **Precision for class 1:** among records predicted as Diabetes, the proportion that are actually labelled Diabetes.
- **Recall for class 1:** among records labelled Diabetes, the proportion identified by the model.
- **F1-score for class 1:** the harmonic mean of precision and recall.
- **Confusion matrix:** counts of true negatives, false positives, false negatives, and true positives.

The confusion matrix is displayed with actual labels in rows and predicted labels in columns.

## 4. Performance analysis

Training and test accuracy are compared to check how the model performs on the records used for fitting and on the held-out records. A higher training score can indicate that a model fits the training data better than unseen data, but one split is not enough to confirm overfitting.

For this dataset split, the test recall for class 1 is 50%. This means that half of the actual Diabetes-class records in the test set were identified. The false-negative count is therefore an important part of the results discussion; accuracy should not be considered on its own.

## Limitations and possible next steps

This is a baseline model evaluated on one train/test split of the supplied dataset. The results may change with a different sample, split, preprocessing assumption, or model. Further analysis could use cross-validation, compare other classifiers, examine decision thresholds, and report additional class-sensitive metrics. Any such work would need independent validation before the model could be considered for a real application.

The project is for educational purposes only and is not a diagnostic or clinical decision-support tool.
