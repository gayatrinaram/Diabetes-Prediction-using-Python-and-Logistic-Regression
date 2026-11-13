# BMI Category Classification with Logistic Regression

A Python and scikit-learn notebook exploring a supervised classification workflow on a diabetes-related dataset. The current notebook uses the `BmiClass` field as its target and predicts four BMI categories: underweight, normal weight, overweight, and obese.

> **Important clarification:** Despite the repository's original name, the implemented target is BMI category, not diabetes diagnosis. The notebook's final mapping of BMI categories to “Diabetic” and “Not Diabetic” is not medically valid. This project must not be used to diagnose diabetes or make healthcare decisions.

## Project overview

The notebook demonstrates a basic end-to-end machine-learning workflow:

1. Load a CSV dataset with pandas.
2. Inspect the dataset and check for missing values.
3. Encode categorical features.
4. prepare the target using BMI-category labels.
5. Split the data into training and test sets.
6. Train a Logistic Regression classifier.
7. Evaluate predictions with accuracy, a confusion matrix, and a classification report.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Google Colab / Jupyter Notebook

## Run the notebook

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gayatrinaram/Diabetes-Prediction-using-Python-and-Logistic-Regression/blob/main/Diabetes_Prediction_using_Python_and_Logistic_Regression_Project.ipynb)

1. Open the notebook in Google Colab.
2. Upload or mount the dataset expected by the notebook.
3. Update the `csv_file` path in the data-loading cell to match your file.
4. Run the notebook from top to bottom and review the evaluation output.

## Repository contents

- `Diabetes_Prediction_using_Python_and_Logistic_Regression_Project.ipynb` — data preparation, model training, and evaluation notebook.

The dataset is not included in this repository. Make sure you have the right to use any dataset you provide.

## Observed baseline and evaluation notes

The notebook's saved run reports approximately **75.19% accuracy** on a 20,000-row test split. However, its classification report shows zero precision and recall for the normal-weight and overweight classes, indicating that accuracy alone gives a misleading picture of performance.

The notebook also reports a Logistic Regression convergence warning. Treat the saved score as an exploratory baseline, not as evidence of a reliable or production-ready model.

## Known limitations to address

- The final BMI-to-diabetes label mapping is incorrect and should be removed.
- Use `BmiClass` consistently as the prediction target and report BMI categories as the model output.
- Fit encoders and preprocessing steps on the training data only, ideally with a scikit-learn Pipeline, to avoid data leakage.
- Scale numeric features and tune model settings to address convergence.
- Evaluate class-wise precision, recall, F1-score, and a confusion matrix; consider stratified splitting and class-imbalance handling.
- Add reproducible dataset instructions and a fixed random seed.

## Responsible use

This is an educational machine-learning exercise. BMI category is not a diagnosis of diabetes, and model predictions should not be used for clinical decisions.

## Author

**Gayatri Naram**  
[GitHub profile](https://github.com/gayatrinaram)
