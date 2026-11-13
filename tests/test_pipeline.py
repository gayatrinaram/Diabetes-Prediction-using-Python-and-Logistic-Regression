import unittest
import pandas as pd
from train import build_pipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_fits_mixed_feature_types(self):
        X = pd.DataFrame({
            "age": [21, 35, 44, 52, 28, 61, 33, 47],
            "activity": ["low", "high", "medium", "low", "high", "medium", "low", "high"],
        })
        y = pd.Series(["Normal", "Overweight", "Obese", "Obese", "Normal", "Obese", "Overweight", "Normal"])
        model = build_pipeline(X)
        model.fit(X, y)
        predictions = model.predict(X.iloc[:2])
        self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
