# BMI Category Classification with Logistic Regression

An educational machine-learning project that classifies tabular records into BMI categories using Python and scikit-learn. It includes a notebook for exploration and reusable scripts for training and batch prediction.

> **Important clarification:** The historical repository name is retained for continuity, but the implemented target is `BmiClass`, not diabetes diagnosis. BMI categories must not be mapped to diabetic or non-diabetic labels. This project is not a medical or diagnostic tool.

## Project workflow

1. Load a CSV containing a `BmiClass` target column.
2. Separate features from the target.
3. Impute missing numeric and categorical values.
4. Scale numeric features and one-hot encode categorical features.
5. Split data into stratified training and test sets.
6. Train a Logistic Regression classifier in a scikit-learn Pipeline.
7. Evaluate accuracy, class-wise precision, recall, F1-score, and a confusion matrix.
8. Save the trained pipeline for future batch predictions.

## Tech stack

Python · pandas · scikit-learn · Matplotlib · joblib · Google Colab

## Run the notebook

[Open in Google Colab](https://colab.research.google.com/github/gayatrinaram/Diabetes-Prediction-using-Python-and-Logistic-Regression/blob/main/Diabetes_Prediction_using_Python_and_Logistic_Regression_Project.ipynb)

The notebook prompts for a CSV or ZIP file containing a `BmiClass` column.

## Run the reusable training script

```bash
git clone https://github.com/gayatrinaram/Diabetes-Prediction-using-Python-and-Logistic-Regression.git
cd Diabetes-Prediction-using-Python-and-Logistic-Regression
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --csv path/to/your_dataset.csv
```

The training script saves the fitted pipeline to `models/bmi_category_pipeline.joblib` by default. The dataset is not included in this repository; provide a CSV you are authorised to use.

## Run batch predictions

Prepare a CSV containing the same feature columns used during training, without the target column:

```bash
python src/predict.py --input path/to/new_records.csv
```

Predictions are written to `predictions.csv` by default.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Repository structure

```text
├── src/
│   ├── train.py       # Preprocessing, training, evaluation and model export
│   └── predict.py     # Batch inference using the saved pipeline
├── tests/
│   └── test_pipeline.py
├── Diabetes_Prediction_using_Python_and_Logistic_Regression_Project.ipynb
├── requirements.txt
└── README.md
```

## Modelling notes

Preprocessing is encapsulated in a Pipeline and fitted using training data only to reduce leakage. Stratified splitting and class-wise metrics are used because accuracy alone can hide poor performance on underrepresented classes. The balanced class-weight setting is a baseline to compare, not a guarantee of better performance. Model quality must be evaluated on the intended dataset before drawing conclusions.

## Responsible use

BMI is not a diagnosis of diabetes or a substitute for professional assessment. Predictions from this educational exercise must not be used for healthcare decisions.

## Author

**Gayatri Naram** · [GitHub profile](https://github.com/gayatrinaram)
