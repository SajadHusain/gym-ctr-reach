"""Numerical checks executable with NumPy and stdlib unittest, without PyTorch."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("diagnostic_metrics",
    Path(__file__).resolve().parents[1]/"ctr_reach_envs/ivp/diagnostic_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class MotionDiagnosticsTest(unittest.TestCase):
    def test_exact_prediction_has_zero_error_and_positive_progress(self):
        r = metrics.motion_metrics([0, 0, 0], [.01, 0, 0], [.002, 0, 0], [.002, 0, 0])
        self.assertAlmostEqual(r["actual_progress_m"], .002)
        self.assertEqual(r["linearization_error_m"], 0.)
        self.assertEqual(r["motion_cosine"], 1.)
        self.assertTrue(r["progress_sign_agrees"])

    def test_wrong_direction_is_detected(self):
        r = metrics.motion_metrics([0, 0, 0], [.01, 0, 0], [-.002, 0, 0], [.002, 0, 0])
        self.assertLess(r["actual_progress_m"], 0)
        self.assertFalse(r["progress_sign_agrees"])
        self.assertAlmostEqual(r["relative_linearization_error"], 2.)
        self.assertEqual(r["motion_cosine"], -1.)

    def test_zero_motion_has_no_direction_or_sign_evidence(self):
        r = metrics.motion_metrics([0, 0, 0], [.01, 0, 0], [0, 0, 0], [0, 0, 0])
        self.assertIsNone(r["motion_cosine"])
        self.assertIsNone(r["progress_sign_agrees"])
        self.assertEqual(r["relative_linearization_error"], 0.)

    def test_invalid_jacobian_retains_actual_motion(self):
        r = metrics.motion_metrics([0, 0, 0], [.01, 0, 0], [.002, 0, 0])
        self.assertAlmostEqual(r["actual_progress_m"], .002)
        self.assertIsNone(r["linearization_error_m"])

    def test_failed_cases_are_counted_and_not_averaged_as_zero_error(self):
        r = dict(metrics.motion_metrics([0, 0, 0], [.01, 0, 0], [.002, 0, 0], [.002, 0, 0]),
                 jacobian_valid=True, failure="")
        summary = metrics.summarize_motion([r, {"jacobian_valid":False, "failure":"solver"}])
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["linearization_error_m"]["count"], 1)

    def test_nonfinite_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.motion_metrics([0, 0, 0], [1, 0, 0], [0, 0, 0], [np.nan, 0, 0])


if __name__ == "__main__":
    unittest.main()
