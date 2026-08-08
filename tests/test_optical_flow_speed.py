import unittest

import cv2
import numpy as np

from optical_flow_speed import OpticalFlowSpeedEstimator


class TestOpticalFlowSpeedEstimator(unittest.TestCase):
    @staticmethod
    def _texture(seed=7):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        return cv2.GaussianBlur(image, (5, 5), 0)

    def test_translation_is_converted_to_metric_speed(self):
        first = self._texture()
        dx_px, dy_px = 8.0, -6.0
        matrix = np.float32([[1, 0, dx_px], [0, 1, dy_px]])
        second = cv2.warpAffine(first, matrix, (first.shape[1], first.shape[0]))
        estimator = OpticalFlowSpeedEstimator(max_dimension=960, min_tracks=20)

        warmup = estimator.update(first, 100.0, 0.20)
        result = estimator.update(second, 102.0, 0.20)

        self.assertFalse(warmup.valid)
        self.assertTrue(result.valid, result.reason)
        # 10 px * 0.20 m/px / 2 s = 1 m/s.
        self.assertAlmostEqual(result.speed_mps, 1.0, delta=0.12)
        self.assertGreaterEqual(result.tracked_points, 20)

    def test_rotation_about_image_center_does_not_look_like_translation(self):
        first = self._texture()
        center = (first.shape[1] / 2.0, first.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, 4.0, 1.0)
        second = cv2.warpAffine(first, matrix, (first.shape[1], first.shape[0]))
        estimator = OpticalFlowSpeedEstimator(max_dimension=960, min_tracks=20)

        estimator.update(first, 200.0, 0.25)
        result = estimator.update(second, 201.0, 0.25)

        self.assertTrue(result.valid, result.reason)
        self.assertLess(result.speed_mps, 0.35)

    def test_missing_timestamp_reports_unavailable_speed(self):
        image = self._texture()
        estimator = OpticalFlowSpeedEstimator()

        estimator.update(image, None, 0.20)
        result = estimator.update(image, None, 0.20)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "timestamp_missing")


if __name__ == "__main__":
    unittest.main()
