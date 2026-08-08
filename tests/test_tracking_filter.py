# -*- coding: utf-8 -*-

import unittest

from tracking_filter import ConstantVelocityKalmanFilter


class TestConstantVelocityKalmanFilter(unittest.TestCase):
    def test_known_control_moves_position_exactly_before_update(self):
        kalman = ConstantVelocityKalmanFilter((100, 200), 2.0, 5.0)
        kalman.predict(10.0, -4.0)
        self.assertEqual(kalman.position, (110, 196))

    def test_repeated_measurements_learn_velocity_and_coast(self):
        kalman = ConstantVelocityKalmanFilter((0, 0), 1.0, 2.0)
        for x in (10, 20, 30, 40):
            kalman.predict()
            kalman.update(x, 0, confidence=1.0)
        before = kalman.position[0]
        self.assertGreater(kalman.velocity[0], 0.0)
        kalman.predict()
        self.assertGreater(kalman.position[0], before)

    def test_reset_clears_learned_velocity(self):
        kalman = ConstantVelocityKalmanFilter((0, 0), 1.0, 2.0)
        kalman.predict()
        kalman.update(20, 0)
        kalman.reset((5, 6))
        self.assertEqual(kalman.position, (5, 6))
        self.assertEqual(kalman.velocity, (0.0, 0.0))

    def test_gain_cap_limits_single_update(self):
        kalman = ConstantVelocityKalmanFilter((0, 0), 100.0, 1.0)
        kalman.predict()
        kalman.update(100, 0, gain_max=0.2)
        self.assertLessEqual(kalman.position[0], 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
