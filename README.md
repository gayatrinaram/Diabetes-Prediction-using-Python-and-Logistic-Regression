# BMI Category Classification with Logistic Regression

An educational machine-learning project that uses Python and scikit-learn to classify records into BMI categories with Logistic Regression.

[**Open the notebook in Google Colab**](https://colab.research.google.com/github/gayatrinaram/Diabetes-Prediction-using-Python-and-Logistic-Regression/blob/main/Diabetes_Prediction_using_Python_and_Logistic_Regression_Project.ipynb)

> **Important:** Despite the repository's historical name, the implemented target is `BmiClass`, not diabetes diagnosis. BMI categories must not be mapped to diabetic or non-diabetic labels. This project is not a medical or diagnostic tool.

## Workflow

1. Upload a CSV or ZIP file containing a `BmiClass` target column.
2. Separate features from the target.
3. Identify numeric and categorical features.
4. Build preprocessing pipelines for imputation, scaling, and one-hot encoding.
5. Split the data into stratified training and test sets.
6. Train a Logistic Regression classifier.
7. Evaluate accuracy, class-wise precision, recall, F1-score, and the confusion matrix.

## Tech stack

- Python
- pandas
- scikit-learn
- Matplotlib
- Google Colab / Jupyter Notebook

## Run it

1. Open the Colab link above.
2. Run the notebook.
3. Upload a CSV or ZIP file containing the dataset.
4. Ensure the dataset has a `BmiClass` column with at least two categories.
5. Review the evaluation metrics and confusion matrix.

The dataset is not included in this repository. Use a dataset you are authorised to access and make sure its target labels and provenance are documented.

## Repository structure

```text
Diabetes-Prediction-using-Python-and-Logistic-Regression/
├── Diabetes_Prediction_using_Python_and_Logistic_Regression_Project.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Modelling notes

The notebook uses a scikit-learn Pipeline so preprocessing is fitted on the training split rather than the full dataset. It also uses stratified splitting and reports class-wise metrics because accuracy alone can hide poor performance on underrepresented classes. The balanced class-weight setting is a baseline to compare, not a guarantee of better performance.

## Responsible use

BMI is not a diagnosis of diabetes or a substitute for professional assessment. Predictions from this educational exercise must not be used for healthcare decisions.

## Author

**Gayatri Naram** · [GitHub](https://github.com/gayatrinaram)
