# -*- coding: utf-8 -*-

import unittest
import os
import tempfile

from localization_policy import (
    choose_visualization_view,
    choose_dem_query_latlon,
    next_search_roi_size,
    resolve_camera_altitude_for_dem,
    load_motion_series,
    select_external_motion_pixels,
    select_failure_growth_px,
    select_inference_altitude,
    select_localizer_ground_truth_rc,
    validate_inference_policy,
    should_run_global_recovery,
    should_reset_search_roi,
)


class TestInferencePolicy(unittest.TestCase):
    def test_altitude_anchor_survives_wrong_high_dem_at_predicted_position(self):
        effective0, clearance0, anchor_alt, anchor_clearance = resolve_camera_altitude_for_dem(
            sensor_altitude_m=1503.0,
            terrain_center_m=1104.0,
            datum_correction_m=26.0,
            use_initial_clearance_anchor=True,
        )
        self.assertAlmostEqual(clearance0, 425.0)
        self.assertAlmostEqual(effective0, 1503.0)

        effective1, clearance1, _, _ = resolve_camera_altitude_for_dem(
            sensor_altitude_m=1550.0,
            terrain_center_m=1589.7034912109375,
            datum_correction_m=26.0,
            use_initial_clearance_anchor=True,
            anchor_sensor_altitude_m=anchor_alt,
            anchor_ground_clearance_m=anchor_clearance,
        )
        self.assertAlmostEqual(clearance1, 472.0)
        self.assertAlmostEqual(effective1 - 1589.7034912109375 + 26.0, 472.0)

    def test_absolute_altitude_mode_still_rejects_negative_clearance(self):
        with self.assertRaises(ValueError):
            resolve_camera_altitude_for_dem(
                sensor_altitude_m=1500.0,
                terrain_center_m=1589.0,
                datum_correction_m=26.0,
                use_initial_clearance_anchor=False,
            )

    def test_kalman_tracking_uses_slow_growth_only_for_short_dropout(self):
        self.assertEqual(
            select_failure_growth_px(
                full_growth_px=500,
                tracked_growth_factor=0.4,
                kalman_tracking_available=True,
                consecutive_failures=2,
                tracked_failure_frames=3,
            ),
            200,
        )
        self.assertEqual(
            select_failure_growth_px(
                full_growth_px=500,
                tracked_growth_factor=0.4,
                kalman_tracking_available=True,
                consecutive_failures=4,
                tracked_failure_frames=3,
            ),
            500,
        )

    def test_global_recovery_waits_until_roi_covers_map(self):
        args = dict(
            benchmark=False,
            enabled=True,
            search_anchor_available=True,
            consecutive_failures=20,
            minimum_failure_frames=3,
            minimum_window_px=15000,
            reference_shape=(22516, 30733),
        )
        self.assertFalse(should_run_global_recovery(current_window_px=14999, **args))
        self.assertTrue(should_run_global_recovery(current_window_px=15000, **args))

    def test_dropout_sequence_grows_slow_then_full_before_global(self):
        current = 2048
        failure_streak = 0
        sizes = []
        while current < 15000:
            failure_streak += 1
            growth = select_failure_growth_px(
                full_growth_px=500,
                tracked_growth_factor=0.4,
                kalman_tracking_available=True,
                consecutive_failures=failure_streak,
                tracked_failure_frames=3,
            )
            current = next_search_roi_size(
                current_px=current,
                base_px=2048,
                maximum_px=30733,
                measurement_accepted=False,
                growth_px=growth,
            )
            sizes.append(current)
        self.assertEqual(sizes[:4], [2248, 2448, 2648, 3148])
        self.assertEqual(sizes[-1], 15148)
        self.assertTrue(
            should_run_global_recovery(
                benchmark=False,
                enabled=True,
                search_anchor_available=True,
                consecutive_failures=failure_streak,
                minimum_failure_frames=3,
                minimum_window_px=15000,
                current_window_px=current,
                reference_shape=(22516, 30733),
            )
        )

    def test_external_motion_uses_east_north_without_ground_truth_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "motion.csv")
            with open(path, "w", encoding="utf-8", newline="") as stream:
                stream.write("image_path,east_m,north_m\nframe.jpg,29.85,14.925\n")
            motion = load_motion_series(path)
            pixels, source = select_external_motion_pixels(
                image_path="C:/frames/frame.jpg",
                external_motion=motion,
                map_resolution_m_per_px=0.2985,
            )
        self.assertAlmostEqual(pixels[0], 100.0)
        self.assertAlmostEqual(pixels[1], -50.0)
        self.assertEqual(source, "external_motion_csv")

    def test_failed_frame_grows_search_window_only_one_step(self):
        next_size = next_search_roi_size(
            current_px=2048,
            base_px=2048,
            maximum_px=15000,
            measurement_accepted=False,
            growth_px=500,
        )
        self.assertEqual(next_size, 2548)

    def test_accepted_frame_resets_search_window(self):
        next_size = next_search_roi_size(
            current_px=7048,
            base_px=2048,
            maximum_px=15000,
            measurement_accepted=True,
            growth_px=500,
        )
        self.assertEqual(next_size, 2048)

    def test_reliable_three_way_resets_roi_even_while_kalman_waits(self):
        reset = should_reset_search_roi(
            measurement_valid=True,
            intersection_mode="abc",
            kalman_action="reacq_wait",
        )
        self.assertTrue(reset)
        self.assertEqual(
            next_search_roi_size(
                current_px=7048,
                base_px=2048,
                maximum_px=15000,
                measurement_accepted=reset,
                growth_px=500,
            ),
            2048,
        )

    def test_two_way_reacq_wait_does_not_reset_roi(self):
        self.assertFalse(
            should_reset_search_roi(
                measurement_valid=True,
                intersection_mode="ab",
                kalman_action="reacq_wait",
            )
        )

    def test_covariance_target_is_rate_limited_and_never_shrinks_on_failure(self):
        self.assertEqual(
            next_search_roi_size(
                current_px=2548,
                base_px=2048,
                maximum_px=15000,
                measurement_accepted=False,
                growth_px=500,
                covariance_target_px=9000,
            ),
            3048,
        )
        self.assertEqual(
            next_search_roi_size(
                current_px=3048,
                base_px=2048,
                maximum_px=15000,
                measurement_accepted=False,
                growth_px=500,
                covariance_target_px=2500,
            ),
            3048,
        )

    def test_visualization_auto_fit_keeps_prediction_and_truth_visible(self):
        center, half_size, source = choose_visualization_view(
            benchmark=False,
            predicted_xy=(1000, 1000),
            ground_truth_xy=(5000, 1000),
            mode="auto_fit",
            base_half_size_px=3000,
            maximum_half_size_px=4000,
            edge_margin_px=300,
        )
        self.assertEqual(center, (3000.0, 1000.0))
        self.assertEqual(half_size, 3000)
        self.assertEqual(source, "evaluator_auto_fit")
        self.assertLessEqual(abs(1000 - center[0]), half_size - 300)
        self.assertLessEqual(abs(5000 - center[0]), half_size - 300)

    def test_visualization_falls_back_to_truth_when_both_do_not_fit(self):
        center, half_size, source = choose_visualization_view(
            benchmark=False,
            predicted_xy=(1000, 1000),
            ground_truth_xy=(20000, 1000),
            mode="auto_fit",
            base_half_size_px=3000,
            maximum_half_size_px=4000,
            edge_margin_px=300,
        )
        self.assertEqual(center, (20000.0, 1000.0))
        self.assertEqual(half_size, 3000)
        self.assertEqual(source, "evaluator_ground_truth_fallback")

    def test_benchmark_uses_oracle_ground_truth(self):
        selected = choose_dem_query_latlon(
            benchmark=True,
            strict_gps_denied=False,
            ground_truth_latlon=(38.6, 34.9),
            search_center_rc=(10, 20),
            rc_to_lon_lat=lambda row, col: (1.0, 2.0),
        )
        self.assertEqual(selected, (38.6, 34.9))

    def test_strict_mode_ignores_future_ground_truth(self):
        converter = lambda row, col: (34.0 + col / 1000.0, 38.0 + row / 1000.0)
        first = choose_dem_query_latlon(
            benchmark=False,
            strict_gps_denied=True,
            ground_truth_latlon=(10.0, 20.0),
            search_center_rc=(100, 200),
            rc_to_lon_lat=converter,
        )
        second = choose_dem_query_latlon(
            benchmark=False,
            strict_gps_denied=True,
            ground_truth_latlon=(-70.0, 140.0),
            search_center_rc=(100, 200),
            rc_to_lon_lat=converter,
        )
        self.assertEqual(first, second)
        self.assertEqual(first, (38.1, 34.2))

    def test_strict_policy_rejects_gps_aids(self):
        with self.assertRaises(ValueError):
            validate_inference_policy(
                benchmark=False,
                strict_gps_denied=True,
                use_exif_motion_search_prior=True,
                use_gps_revert=False,
            )
        with self.assertRaises(ValueError):
            validate_inference_policy(
                benchmark=False,
                strict_gps_denied=True,
                use_exif_motion_search_prior=False,
                use_gps_revert=True,
            )

    def test_strict_t_after_zero_hides_ground_truth_from_localizer(self):
        for future_ground_truth in ((100, 200), (9000, 12000), None):
            self.assertIsNone(
                select_localizer_ground_truth_rc(
                    benchmark=False,
                    strict_gps_denied=True,
                    tracking_seeded=True,
                    ground_truth_rc=future_ground_truth,
                )
            )
        self.assertEqual(
            select_localizer_ground_truth_rc(
                benchmark=False,
                strict_gps_denied=True,
                tracking_seeded=False,
                ground_truth_rc=(100, 200),
            ),
            (100, 200),
        )

    def test_initial_hold_ignores_future_gnss_altitude(self):
        first = select_inference_altitude(
            benchmark=False, strict_gps_denied=True, altitude_source="initial_hold",
            observed_exif_altitude_m=1537.0, initial_altitude_m=1537.0,
            image_path="a.jpg",
        )
        later = select_inference_altitude(
            benchmark=False, strict_gps_denied=True, altitude_source="initial_hold",
            observed_exif_altitude_m=1666.0, initial_altitude_m=1537.0,
            image_path="b.jpg",
        )
        self.assertEqual(first, later)
        self.assertEqual(later, (1537.0, "initial_altitude_hold"))

    def test_external_altitude_does_not_require_exif_altitude(self):
        selected = select_inference_altitude(
            benchmark=False, strict_gps_denied=True, altitude_source="external_csv",
            observed_exif_altitude_m=None, initial_altitude_m=None,
            image_path="C:/frames/b.jpg",
            external_altitudes={"b.jpg": 1550.5},
        )
        self.assertEqual(selected, (1550.5, "external_altitude_csv"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
