"""Load a trained BMI-category pipeline and predict categories for new CSV rows."""
import argparse
import joblib
import pandas as pd


def predict(input_csv: str, model_path: str, output_csv: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    features = pd.read_csv(input_csv)
    predictions = model.predict(features)
    result = features.copy()
    result["predicted_bmi_category"] = predictions
    result.to_csv(output_csv, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing feature columns only")
    parser.add_argument("--model", default="models/bmi_category_pipeline.joblib")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    output = predict(args.input, args.model, args.output)
    print(output.head())
    print(f"Saved predictions to {args.output}")
