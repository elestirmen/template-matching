"""Qt integration: controls, real worker lifecycle, image resizing and telemetry.

Run with the Windows visual_navigation environment. The pipeline is replaced
only in worker tests; no user datasets, models or result files are touched.
"""

import os
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
try:
    from PyQt5.QtWidgets import QApplication
    import konum_ui_qt as ui
except ImportError as exc:
    raise unittest.SkipTest(f"Optional desktop dependencies unavailable: {exc}") from exc


class TestMissionUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = ui.MainWindow()

    def tearDown(self):
        if self.window._worker and self.window._worker.isRunning():
            self.window._worker.stop()
            self.assertTrue(self.window._worker.wait(3000))
        self.app.processEvents()
        self.window.close()

    def wait_until(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertTrue(predicate())

    def test_metrics_keep_speed_channels_separate_and_missing_values_unavailable(self):
        panel = self.window._panel
        panel.update_metrics({"spd_kmh": 32.5, "of_spd_valid": False, "of_spd_kmh": 99.0})
        self.assertEqual(panel.c_spd._val.text(), "32.5 km/h")
        self.assertEqual(panel.c_of_spd._val.text(), "—")
        self.assertEqual(panel.c_err._val.text(), "—")
        panel.update_metrics({"spd_kmh": 32.5, "of_spd_valid": True, "of_spd_kmh": 28.0})
        self.assertEqual(panel.c_of_spd._val.text(), "28.0 km/h")

    def test_images_rescale_from_source_after_panel_resize(self):
        view = self.window._map
        self.window.show()
        self.window.resize(1180, 760)
        self.app.processEvents()
        view.update_frame(np.zeros((100, 200, 3), dtype=np.uint8))
        self.window.resize(1480, 920)
        self.app.processEvents()
        self.assertEqual(view._source.width(), 200)
        self.assertEqual(view._source.height(), 100)
        self.assertLessEqual(view.pixmap().width(), view.width())
        self.assertLessEqual(view.pixmap().height(), view.height())
        self.assertAlmostEqual(view.pixmap().width() / view.pixmap().height(), 2, delta=0.02)

    def test_worker_delivers_frames_without_mutating_pipeline_metadata(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        metadata = {"frame": 1, "total": 2, "vis_crop": frame, "vis_model": frame[:, :, 0]}

        def pipeline(callbacks):
            self.assertFalse(callbacks["ui_state_update"]()["kalman"])
            callbacks["on_frame"](frame, metadata)

        self.window._panel._toggles["kalman"].setChecked(False)
        with patch.object(ui._tm, "run_pipeline", pipeline):
            self.window._on_start()
            self.wait_until(lambda: self.window._panel.btn_start.isEnabled())
        self.assertIn("vis_crop", metadata)
        self.assertIsNotNone(self.window._map._source)
        self.assertIsNotNone(self.window._model_panel._img._source)
        self.assertEqual(self.window._panel.progress.value(), 50)
        self.assertEqual(self.window.status_pill.text(), "TAMAMLANDI")

    def test_preprocessing_choice_reaches_pipeline_and_changes_live(self):
        seen = []

        def pipeline(callbacks):
            # Hat, secilen on isleme modunu her karede yeniden okur.
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and len(seen) < 2:
                mode = callbacks["ui_state_update"]()["model_norm"]
                if not seen or seen[-1] != mode:
                    seen.append(mode)
                time.sleep(0.01)

        self.window._panel.combo_norm.setCurrentText("CLAHE")
        with patch.object(ui._tm, "run_pipeline", pipeline):
            self.window._on_start()
            self.wait_until(lambda: seen == ["CLAHE"])
            self.window._panel.combo_norm.setCurrentText("EDGE")
            self.wait_until(lambda: self.window._panel.btn_start.isEnabled())
        self.assertEqual(seen, ["CLAHE", "EDGE"])

    def test_multi_candidate_toggle_and_k_reach_pipeline_and_change_live(self):
        seen = []

        def pipeline(callbacks):
            # Hat her karede UI durumunu yeniden okur; degisiklikler canli yansir.
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and len(seen) < 2:
                state = callbacks["ui_state_update"]()
                current = (state["multi_candidate"], state["pyramid_top_k"])
                if not seen or seen[-1] != current:
                    seen.append(current)
                time.sleep(0.01)

        self.window._panel._toggles["multi_candidate"].setChecked(True)
        self.window._panel.combo_top_k.setCurrentText("3")
        with patch.object(ui._tm, "run_pipeline", pipeline):
            self.window._on_start()
            self.wait_until(lambda: seen == [(True, "3")])
            self.window._panel._toggles["multi_candidate"].setChecked(False)
            self.wait_until(lambda: self.window._panel.btn_start.isEnabled())
        self.assertEqual(seen, [(True, "3"), (False, "3")])

    def test_multi_candidate_toggle_defaults_to_run_cfg_and_is_a_method_control(self):
        button = self.window._panel._toggles["multi_candidate"]
        self.assertEqual(
            button.isChecked(), bool(ui.RUN_CFG.get("PYRAMID_MULTI_CANDIDATE", False))
        )
        # Arama davranisini degistirir; salt gorunum toggle'i degildir.
        self.assertEqual(button.property("controlRole"), "method")
        combo = self.window._panel.combo_top_k
        self.assertEqual(
            [combo.itemText(index) for index in range(combo.count())],
            list(ui._tm.PYRAMID_TOP_K_CHOICES),
        )
        self.assertEqual(combo.currentText(), str(int(ui.RUN_CFG.get("PYRAMID_TOP_K", 5))))

    def test_preprocessing_combo_offers_every_supported_mode(self):
        combo = self.window._panel.combo_norm
        modes = [combo.itemText(index) for index in range(combo.count())]
        self.assertEqual(sorted(modes), sorted(ui.MODEL_INPUT_NORM_MODES))
        self.assertEqual(combo.currentText(), ui.MODEL_INPUT_NORM)

    def test_pipeline_failure_is_not_reported_as_completion(self):
        with patch.object(ui._tm, "run_pipeline", side_effect=RuntimeError("test failure")):
            self.window._on_start()
            self.wait_until(lambda: self.window._panel.btn_start.isEnabled())
        self.assertEqual(self.window.status_pill.text(), "HATA")
        self.assertIn("test failure", self.window._panel.log_box.toPlainText())

    def test_stop_and_close_wait_for_actual_worker_exit(self):
        def pipeline(callbacks):
            while not callbacks["stop_check"]():
                time.sleep(0.01)
            time.sleep(0.1)

        self.window.show()
        with patch.object(ui._tm, "run_pipeline", pipeline):
            self.window._on_start()
            self.window.close()
            self.assertTrue(self.window.isVisible())
            self.assertFalse(self.window._panel.btn_start.isEnabled())
            self.assertEqual(self.window.status_pill.text(), "DURDURULUYOR")
            self.wait_until(lambda: not self.window.isVisible())
        self.assertEqual(self.window.status_pill.text(), "DURDURULDU")
        self.assertFalse(self.window._worker.isRunning())


if __name__ == "__main__":
    unittest.main()
