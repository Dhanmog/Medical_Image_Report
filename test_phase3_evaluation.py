import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_framework import Phase3EvaluationEngine


class TestPhase3Evaluation(unittest.TestCase):
    """Unit and Integration Tests for CHEST AI Phase 3 Evaluation Framework."""

    def setUp(self):
        self.engine = Phase3EvaluationEngine()

    def test_01_dataset_loading(self):
        """Verify that 10 synthetic benchmark cases are loaded correctly."""
        self.assertEqual(len(self.engine.dataset), 10)
        for case in self.engine.dataset:
            self.assertIn("id", case)
            self.assertIn("title", case)
            self.assertIn("raw_transformer_report", case)

    def test_02_single_case_evaluation(self):
        """Verify evaluation metrics computation on case_01 (Cardiomegaly & Effusion)."""
        case1 = self.engine.dataset[0]
        res = self.engine.evaluate_case(case1)
        self.assertEqual(res["case_id"], "case_01")
        self.assertIn("metrics", res)
        metrics = res["metrics"]
        self.assertGreaterEqual(metrics["precision"], 0.0)
        self.assertGreaterEqual(metrics["recall"], 0.0)
        self.assertGreaterEqual(metrics["f1_score"], 0.0)
        self.assertEqual(metrics["hallucination_count"], 0)

    def test_03_full_benchmark_run(self):
        """Verify full benchmark execution and aggregate metric generation."""
        summary = self.engine.run_full_benchmark()
        self.assertIn("aggregate_metrics", summary)
        agg = summary["aggregate_metrics"]
        self.assertIn("mean_precision", agg)
        self.assertIn("mean_recall", agg)
        self.assertIn("mean_f1_score", agg)
        self.assertIn("mean_laterality_accuracy", agg)
        self.assertIn("mean_device_accuracy", agg)
        self.assertEqual(agg["total_hallucinations_detected"], 0)


if __name__ == "__main__":
    unittest.main()
