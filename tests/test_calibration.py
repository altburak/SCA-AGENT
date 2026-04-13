"""
Tests for sca.calibration — ConfidenceCalibrator
"""
from __future__ import annotations
import unittest
from sca.calibration import ConfidenceCalibrator, MIN_CALIBRATION_SAMPLES


class TestConfidenceCalibrator(unittest.TestCase):
    def test_identity_before_calibration(self):
        cal = ConfidenceCalibrator()
        self.assertAlmostEqual(cal.apply(0.7), 0.7, places=6)

    def test_is_fitted_false_before_calibration(self):
        cal = ConfidenceCalibrator()
        self.assertFalse(cal.is_fitted)

    def test_is_fitted_true_after_calibration(self):
        cal = ConfidenceCalibrator()
        cal.calibrate([0.1, 0.3, 0.5, 0.7, 0.9], [0.05, 0.25, 0.55, 0.75, 0.95])
        self.assertTrue(cal.is_fitted)

    def test_calibrate_insufficient_samples_stays_identity(self):
        cal = ConfidenceCalibrator()
        cal.calibrate([0.5, 0.8], [0.4, 0.75])
        self.assertFalse(cal.is_fitted)
        self.assertAlmostEqual(cal.apply(0.5), 0.5, places=6)

    def test_calibrate_mismatched_lengths_raises(self):
        cal = ConfidenceCalibrator()
        with self.assertRaises(ValueError):
            cal.calibrate([0.1, 0.5, 0.9], [0.1, 0.5])

    def test_apply_returns_float_in_range(self):
        cal = ConfidenceCalibrator()
        cal.calibrate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.15, 0.45, 0.65, 0.85, 1.0])
        for raw in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = cal.apply(raw)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_apply_clips_negative_input(self):
        cal = ConfidenceCalibrator()
        self.assertAlmostEqual(cal.apply(-0.5), 0.0, places=6)

    def test_apply_clips_above_one(self):
        cal = ConfidenceCalibrator()
        self.assertAlmostEqual(cal.apply(1.5), 1.0, places=6)

    def test_apply_batch_correct_length(self):
        cal = ConfidenceCalibrator()
        cal.calibrate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.1, 0.5, 0.6, 0.9, 1.0])
        scores = cal.apply_batch([0.2, 0.5, 0.8])
        self.assertEqual(len(scores), 3)

    def test_apply_batch_identity_without_fit(self):
        cal = ConfidenceCalibrator()
        scores = cal.apply_batch([0.3, 0.6, 0.9])
        for got, expected in zip(scores, [0.3, 0.6, 0.9]):
            self.assertAlmostEqual(got, expected, places=6)

    def test_reset_clears_calibration(self):
        cal = ConfidenceCalibrator()
        cal.calibrate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.1, 0.5, 0.6, 0.9, 1.0])
        self.assertTrue(cal.is_fitted)
        cal.reset()
        self.assertFalse(cal.is_fitted)
        self.assertAlmostEqual(cal.apply(0.7), 0.7, places=6)

    def test_isotonic_regression_monotone(self):
        cal = ConfidenceCalibrator()
        cal.calibrate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.18, 0.42, 0.61, 0.83, 1.0])
        inputs = [0.1, 0.3, 0.5, 0.7, 0.9]
        outputs = cal.apply_batch(inputs)
        for i in range(len(outputs) - 1):
            self.assertLessEqual(outputs[i], outputs[i + 1] + 1e-9)

if __name__ == "__main__":
    unittest.main()