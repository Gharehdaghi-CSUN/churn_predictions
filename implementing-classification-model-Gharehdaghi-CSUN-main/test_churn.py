import unittest
import pandas as pd
import numpy as np
from churn_prediction import load_data, preprocess_data, train_models, evaluate_models

class TestChurnPrediction(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.data = load_data("data/customer_churn.csv")
        self.X, self.y = preprocess_data(self.data)
        self.models_dict, self.X_train, self.X_test, self.y_train, self.y_test = train_models(self.X, self.y)

    def test_load_data(self):
        """Test if data loading works correctly."""
        data = load_data("data/customer_churn.csv")
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue("churn" in data.columns)
        self.assertEqual(data.shape[1], 5)  # 4 features + 1 target

    def test_preprocess_data(self):
        """Test if preprocessing returns correct shapes."""
        X, y = preprocess_data(self.data)
        self.assertEqual(X.shape[0], self.data.shape[0])
        self.assertEqual(X.shape[1], 5)  # 3 numerical + 2 dummy variables for contract_type
        self.assertEqual(len(y), self.data.shape[0])

    def test_train_models(self):
        """Test if all models can be trained."""
        models_dict, X_train, X_test, y_train, y_test = train_models(self.X, self.y)
        self.assertIn("Logistic Regression", models_dict)
        self.assertIn("Decision Tree", models_dict)
        self.assertIn("Random Forest", models_dict)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_test)

    def test_evaluate_models(self):
        """Test if evaluation metrics are reasonable."""
        results = evaluate_models(self.models_dict, self.X_train, self.X_test, self.y_train, self.y_test)
        self.assertIsInstance(results, pd.DataFrame)
        for metric in ["Training Accuracy", "Testing Accuracy", "AUROC Score"]:
            self.assertIn(metric, results.columns)
        for score in results["AUROC Score"]:
            self.assertGreaterEqual(float(score), 0)
            self.assertLessEqual(float(score), 1)

if __name__ == "__main__":
    unittest.main()
