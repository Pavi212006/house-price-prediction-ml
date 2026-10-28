"""Offline unit tests for the housing model helpers."""

import math
import unittest

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house_price_prediction.modeling import build_candidate_models, calculate_metrics


class CalculateMetricsTests(unittest.TestCase):
    def test_metrics_for_simple_example(self):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.0, 2.0, 4.0]
        result = calculate_metrics(actual, predicted)

        self.assertAlmostEqual(result["mae"], 1 / 3)
        self.assertAlmostEqual(result["rmse"], math.sqrt(1 / 3))
        self.assertGreaterEqual(result["r2"], 0.0)
        self.assertLessEqual(result["r2"], 1.0)

    def test_empty_values_raise(self):
        with self.assertRaises(ValueError):
            calculate_metrics([], [])

    def test_different_lengths_raise(self):
        with self.assertRaises(ValueError):
            calculate_metrics([1.0, 2.0], [1.0])

    def test_non_finite_values_raise(self):
        with self.assertRaises(ValueError):
            calculate_metrics([1.0, np.nan], [1.0, 2.0])


class CandidateModelTests(unittest.TestCase):
    def test_candidate_models_have_expected_names(self):
        self.assertEqual(set(build_candidate_models()), {"Linear Regression", "Random Forest"})


if __name__ == "__main__":
    unittest.main()
