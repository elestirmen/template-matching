# -*- coding: utf-8 -*-

import unittest

from run_localization import build_config


class TestRunModeConfig(unittest.TestCase):
    def test_simulation_forces_strict_gps_denied_guards(self):
        config = build_config(
            mode="simulation",
            extra_config={"USE_GPS_REVERT": True, "USE_EXIF_MOTION_SEARCH_PRIOR": True},
        )
        self.assertFalse(config["BENCHMARK"])
        self.assertTrue(config["STRICT_GPS_DENIED_INFERENCE"])
        self.assertFalse(config["USE_GPS_REVERT"])
        self.assertFalse(config["USE_EXIF_MOTION_SEARCH_PRIOR"])
        self.assertEqual(config["ALTITUDE_SOURCE"], "exif_altitude_proxy")

    def test_benchmark_is_explicit_oracle_mode(self):
        config = build_config(mode="benchmark", max_frames=12, sha256_assets=True)
        self.assertTrue(config["BENCHMARK"])
        self.assertEqual(config["MAX_FRAMES"], 12)
        self.assertEqual(config["ASSET_HASH_MODE"], "sha256")
        self.assertEqual(config["ALTITUDE_SOURCE"], "exif_altitude_proxy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
