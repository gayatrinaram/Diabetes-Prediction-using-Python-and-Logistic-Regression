"""Train a BMI-category Logistic Regression baseline from a CSV dataset.

Expected target column: BmiClass. This is not a diabetes prediction model.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([
        ("preprocessor", prep),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train(csv_path: str, target: str = "BmiClass", output: str = "models/bmi_category_pipeline.joblib"):
    data = pd.read_csv(csv_path)
    if target not in data.columns:
        raise ValueError(f"Required target column '{target}' not found.")
    data = data.dropna(subset=[target]).copy()
    X = data.drop(columns=[target])
    y = data[target].astype(str)
    if y.nunique() < 2:
        raise ValueError("At least two target classes are required.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    print(f"Saved model to: {destination}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to a CSV containing BmiClass")
    parser.add_argument("--target", default="BmiClass")
    parser.add_argument("--output", default="models/bmi_category_pipeline.joblib")
    args = parser.parse_args()
    train(args.csv, args.target, args.output)
