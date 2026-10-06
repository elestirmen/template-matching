# -*- coding: utf-8 -*-
"""
Ana betikteki saf (yan etkisiz) fonksiyonlar icin birim testleri.

Bu testler PRODUKSIYON kodunu DEGISTIRMEZ; ana betigi (monolit) bir modul gibi
yukleyip yalnizca matematik/geometri/yardimci fonksiyonlarini dogrular.

Calistirma (proje bagimliliklarinin kurulu oldugu ortamda):
    # conda ortami ornegi:
    python -m unittest discover -s tests -v
    # ya da dogrudan:
    python -m unittest tests.test_core_functions -v

Not: Ana modulun yuklenmesi cv2 / osgeo / tensorflow gibi agir bagimliliklari
ice aktarir. Bunlardan biri yoksa testler otomatik olarak ATLANIR (skip), hata
vermez.
"""

import os
import math
import importlib.util
import unittest

# ---------------------------------------------------------------------------
# Ana betigi (monolit) bir modul olarak yukle.
# konum_ui_qt.py ile ayni yaklasim: dosya yolundan spec olusturup exec et.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_HERE)
_CORE_PY = os.path.join(
    _PROJECT_DIR,
    "template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py",
)

tm = None
_IMPORT_ERROR = None
try:
    import numpy as np  # noqa: F401  (testlerde kullanilir)
    _spec = importlib.util.spec_from_file_location("tm_core_under_test", _CORE_PY)
    tm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tm)
except Exception as exc:  # pragma: no cover - ortam bagimliligi
    tm = None  # yarim yuklenen modul testleri hataya dusurmesin; skipIf ile atlansin
    _IMPORT_ERROR = exc


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi (bagimlilik eksik olabilir): {_IMPORT_ERROR}")
class TestIntersection(unittest.TestCase):
    """intersection(a, b): (x, y, w, h) dortgen kesisimi; kesisim yoksa ()."""

    def test_partial_overlap(self):
        self.assertEqual(tm.intersection((0, 0, 10, 10), (5, 5, 10, 10)), (5, 5, 5, 5))

    def test_identical_boxes(self):
        self.assertEqual(tm.intersection((0, 0, 10, 10), (0, 0, 10, 10)), (0, 0, 10, 10))

    def test_no_overlap(self):
        self.assertEqual(tm.intersection((0, 0, 10, 10), (20, 20, 5, 5)), ())

    def test_edge_touch_is_not_intersection(self):
        # Kenardan temas (sifir alan) gercek kesisim sayilmaz.
        self.assertEqual(tm.intersection((0, 0, 10, 10), (10, 5, 10, 10)), ())

    def test_symmetry(self):
        a, b = (2, 3, 8, 8), (4, 1, 5, 9)
        self.assertEqual(tm.intersection(a, b), tm.intersection(b, a))


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestHaversine(unittest.TestCase):
    """haversine_distance(lat1, lon1, lat2, lon2) -> km."""

    def test_same_point_is_zero(self):
        self.assertAlmostEqual(tm.haversine_distance(38.6, 34.9, 38.6, 34.9), 0.0, places=9)

    def test_one_degree_longitude_at_equator(self):
        # Ekvatorda 1 derece boylam ~ 111.195 km (R=6371 km).
        d = tm.haversine_distance(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(d, 111.195, delta=0.01)

    def test_symmetry(self):
        d1 = tm.haversine_distance(38.60, 34.90, 38.65, 34.95)
        d2 = tm.haversine_distance(38.65, 34.95, 38.60, 34.90)
        self.assertAlmostEqual(d1, d2, places=9)


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestQuickDistance(unittest.TestCase):

    def test_identical_zero(self):
        self.assertEqual(tm.quick_distance(38.6, 34.9, 38.6, 34.9), 0.0)

    def test_positive_for_distinct(self):
        self.assertGreater(tm.quick_distance(38.60, 34.90, 38.61, 34.91), 0.0)

    def test_utm_identical_near_zero(self):
        self.assertAlmostEqual(tm.quick_distance_utm(38.6, 34.9, 38.6, 34.9), 0.0, places=4)

    def test_utm_matches_haversine_for_small_separation(self):
        # Kucuk ayrim icin UTM (metre) ~ haversine (km*1000); %5 toleransla.
        hav_m = tm.haversine_distance(38.60, 34.90, 38.61, 34.90) * 1000.0
        utm_m = tm.quick_distance_utm(38.60, 34.90, 38.61, 34.90)
        self.assertAlmostEqual(utm_m, hav_m, delta=hav_m * 0.05)


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestLatLonToUtm(unittest.TestCase):

    def test_zone_for_turkey(self):
        e, n, zone, hemi = tm.latlon_to_utm(38.6, 34.9)
        self.assertEqual(zone, 36)
        self.assertEqual(hemi, "N")
        self.assertTrue(100000 < e < 900000, f"easting beklenen aralikta degil: {e}")
        self.assertGreater(n, 0)

    def test_zone_for_west_longitude(self):
        _, _, zone, _ = tm.latlon_to_utm(34.0, -120.0)
        self.assertEqual(zone, 11)

    def test_southern_hemisphere(self):
        _, _, _, hemi = tm.latlon_to_utm(-33.9, 18.4)
        self.assertEqual(hemi, "S")


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestErrorMetrics(unittest.TestCase):

    def test_rmse(self):
        errors = np.array([3.0, 4.0])
        self.assertAlmostEqual(tm.rmse(errors), math.sqrt(12.5), places=9)

    def test_rmse_zero(self):
        self.assertAlmostEqual(tm.rmse(np.zeros(5)), 0.0, places=9)

    def test_mae(self):
        self.assertAlmostEqual(tm.mae(np.array([-3.0, 3.0])), 3.0, places=9)

    def test_standart_sapma(self):
        data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
        self.assertAlmostEqual(tm.standart_sapma(data), 2.0, places=9)


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestFileHelpers(unittest.TestCase):

    def test_normalize_ext_set(self):
        out = tm._normalize_ext_set(["JPG", ".png", " .Tif ", "", 5, None])
        self.assertEqual(out, {".jpg", ".png", ".tif"})

    def test_normalize_ext_empty(self):
        self.assertEqual(tm._normalize_ext_set([]), set())

    def test_ext_allowed_true(self):
        self.assertTrue(tm._ext_allowed("foo/bar.JPG", {".jpg"}))

    def test_ext_allowed_false(self):
        self.assertFalse(tm._ext_allowed("foo/bar.txt", {".jpg"}))

    def test_ext_allowed_empty_means_all(self):
        self.assertTrue(tm._ext_allowed("foo/bar.txt", set()))


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestToFloatRatio(unittest.TestCase):

    def test_ratio_tuple(self):
        self.assertAlmostEqual(tm._to_float_ratio((10, 2)), 5.0, places=9)

    def test_zero_denominator_treated_as_one(self):
        self.assertAlmostEqual(tm._to_float_ratio((5, 0)), 5.0, places=9)

    def test_scalar(self):
        self.assertAlmostEqual(tm._to_float_ratio(3), 3.0, places=9)

    def test_invalid_returns_none(self):
        self.assertIsNone(tm._to_float_ratio("abc"))
        self.assertIsNone(tm._to_float_ratio(None))


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestDmsConversion(unittest.TestCase):
    """conversion(yon, coord): DMS -> ondalik derece."""

    def test_north(self):
        self.assertAlmostEqual(tm.conversion("N", (30, 0, 0)), 30.0, places=9)

    def test_south_negative(self):
        self.assertAlmostEqual(tm.conversion("S", (30, 30, 0)), -30.5, places=9)

    def test_seconds(self):
        self.assertAlmostEqual(tm.conversion("N", (0, 0, 3600)), 1.0, places=9)

    def test_west_negative(self):
        self.assertAlmostEqual(tm.conversion("W", (10, 0, 0)), -10.0, places=9)


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestGeometryHelpers(unittest.TestCase):

    def test_largest_rotated_rect_zero_angle(self):
        w, h = tm.largest_rotated_rect(100, 100, 0.0)
        self.assertAlmostEqual(w, 100.0, places=6)
        self.assertAlmostEqual(h, 100.0, places=6)

    def test_largest_rotated_rect_positive_and_bounded(self):
        # 30 derece donus: ic dortgen pozitif ve sinirlayici kutudan kucuk olmali.
        ang = math.radians(30)
        rw, rh = tm.largest_rotated_rect(200, 100, ang)
        self.assertGreater(rw, 0)
        self.assertGreater(rh, 0)
        bb_w = 200 * math.cos(ang) + 100 * math.sin(ang)
        bb_h = 200 * math.sin(ang) + 100 * math.cos(ang)
        self.assertLessEqual(rw, bb_w + 1e-6)
        self.assertLessEqual(rh, bb_h + 1e-6)

    def test_calculate_coordinates_zero_offset(self):
        lat, lon = tm.calculate_coordinates(38.6, 34.9, 0.0, 0.0)
        self.assertAlmostEqual(lat, 38.6, places=9)
        self.assertAlmostEqual(lon, 34.9, places=9)

    def test_calculate_coordinates_north_increases_lat(self):
        lat, _ = tm.calculate_coordinates(38.6, 34.9, 1000.0, 0.0)
        self.assertGreater(lat, 38.6)

    def test_find_corner_coordinates_ordering(self):
        (lat1, lon1), (lat2, lon2) = tm.find_corner_coordinates(38.6, 34.9, 100, 0.30)
        # Sol-ust kose merkezden kucuk, sag-alt kose merkezden buyuk olmali.
        self.assertLess(lat1, 38.6)
        self.assertGreater(lat2, 38.6)
        self.assertLess(lon1, 34.9)
        self.assertGreater(lon2, 34.9)


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestModelInputPreprocessing(unittest.TestCase):
    """Anlik goruntu on isleme: normalizasyon modlari ve model girdisi hazirligi."""

    def setUp(self):
        rng = np.random.default_rng(0)
        # Dusuk kontrastli, parlaklik gradyanli gri yama (esitlemenin etkisi olculebilsin).
        base = np.linspace(60, 120, 64, dtype=np.float32)
        self.patch = np.clip(
            base[None, :] + rng.normal(0.0, 3.0, (64, 64)), 0, 255
        ).astype(np.uint8)

    def test_ham_returns_same_pixels(self):
        out = tm.apply_model_input_norm(self.patch, "HAM")
        self.assertTrue(np.array_equal(out, self.patch))

    def test_histeq_widens_dynamic_range(self):
        out = tm.apply_model_input_norm(self.patch, "HISTEQ")
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(int(out.max()) - int(out.min()),
                           int(self.patch.max()) - int(self.patch.min()))

    def test_clahe_and_edge_return_uint8_same_shape(self):
        for mode in ("CLAHE", "EDGE"):
            out = tm.apply_model_input_norm(self.patch, mode)
            self.assertEqual(out.shape, self.patch.shape, mode)
            self.assertEqual(out.dtype, np.uint8, mode)

    def test_edge_of_flat_patch_is_zero(self):
        flat = np.full((32, 32), 100, dtype=np.uint8)
        self.assertEqual(int(tm.apply_model_input_norm(flat, "EDGE").max()), 0)

    def test_color_input_is_converted_to_gray(self):
        color = np.dstack([self.patch] * 3)
        out = tm.apply_model_input_norm(color, "HAM")
        self.assertEqual(out.ndim, 2)

    def test_unknown_mode_falls_back_to_configured_default(self):
        self.assertEqual(tm.resolve_model_input_norm("YOK"), tm.MODEL_INPUT_NORM)
        self.assertEqual(tm.resolve_model_input_norm(None), tm.MODEL_INPUT_NORM)
        self.assertEqual(tm.resolve_model_input_norm("clahe"), "CLAHE")

    def test_next_mode_cycles_through_all_modes(self):
        mode = tm.MODEL_INPUT_NORM_MODES[0]
        seen = []
        for _ in range(len(tm.MODEL_INPUT_NORM_MODES)):
            mode = tm.next_model_input_norm(mode)
            seen.append(mode)
        self.assertEqual(sorted(seen), sorted(tm.MODEL_INPUT_NORM_MODES))

    def test_prepare_model_input_shape_and_range(self):
        out = tm.prepare_model_input(self.patch, mode="HISTEQ", patch_size=32)
        self.assertEqual(out.shape, (32, 32))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), -1.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_prepare_model_input_matches_previous_pipeline_defaults(self):
        # Onceki davranis: resize(NEAREST) -> equalizeHist -> (x-127.5)/127.5
        import cv2
        expected = cv2.resize(self.patch, (32, 32), interpolation=cv2.INTER_NEAREST)
        expected = (cv2.equalizeHist(expected).astype(np.float32) - 127.5) / 127.5
        out = tm.prepare_model_input(
            self.patch, mode="HISTEQ", before_resize=False, interp="NEAREST", patch_size=32
        )
        self.assertTrue(np.allclose(out, expected))

    def test_norm_before_resize_differs_from_after_resize(self):
        before = tm.prepare_model_input(self.patch, mode="HISTEQ", before_resize=True, patch_size=16)
        after = tm.prepare_model_input(self.patch, mode="HISTEQ", before_resize=False, patch_size=16)
        self.assertFalse(np.allclose(before, after))

    def test_ham_mode_skips_equalization(self):
        raw = tm.prepare_model_input(self.patch, mode="HAM", patch_size=32)
        eq = tm.prepare_model_input(self.patch, mode="HISTEQ", patch_size=32)
        self.assertFalse(np.allclose(raw, eq))


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestRuntimeNormButton(unittest.TestCase):
    """Runtime panelindeki "On Isleme" butonu: ac/kapa degil, sirayla mod degistirir."""

    def setUp(self):
        import cv2
        self.cv2 = cv2
        self.buttons = tm._build_runtime_buttons()
        self.state = {
            "trajectory": True, "inner_frame": False, "roi_frame": True,
            "tm_boxes": True, "model_norm": tm.MODEL_INPUT_NORM,
            "_panel_collapsed": False, "_hover_key": None,
        }
        self.button = next(b for b in self.buttons if b["key"] == "model_norm")

    def _click_center(self):
        bx, by, bw, bh = self.button["rect"]
        tm._runtime_buttons_mouse_cb(
            self.cv2.EVENT_LBUTTONDOWN, bx + bw // 2, by + bh // 2, 0,
            {"state": self.state, "buttons": self.buttons},
        )

    def test_click_cycles_modes_instead_of_toggling_bool(self):
        img = np.zeros((960, 1280, 3), dtype=np.uint8)
        tm._draw_runtime_buttons(img, self.state, self.buttons, display_size=(1280, 960))
        seen = []
        for _ in range(len(tm.MODEL_INPUT_NORM_MODES)):
            self._click_center()
            seen.append(self.state["model_norm"])
        self.assertEqual(sorted(seen), sorted(tm.MODEL_INPUT_NORM_MODES))
        self.assertEqual(self.state["model_norm"], tm.MODEL_INPUT_NORM)

    def test_panel_draws_with_every_mode(self):
        for mode in tm.MODEL_INPUT_NORM_MODES:
            img = np.zeros((960, 1280, 3), dtype=np.uint8)
            self.state["model_norm"] = mode
            tm._draw_runtime_buttons(img, self.state, self.buttons, display_size=(1280, 960))
            self.assertTrue(img.any(), mode)


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi: {_IMPORT_ERROR}")
class TestPyramidUiBridge(unittest.TestCase):
    """UI toggle/secimlerinin RUN_CFG'ye yansimasi ve K dongusu."""

    def setUp(self):
        self._saved = {
            key: tm.RUN_CFG[key]
            for key in ("PYRAMID_MULTI_CANDIDATE", "PYRAMID_TOP_K", "PYRAMID_NMS_RADIUS_PX")
        }

    def tearDown(self):
        tm.RUN_CFG.update(self._saved)

    def test_toggle_and_k_reach_run_cfg(self):
        tm.apply_pyramid_ui_state({"multi_candidate": True, "pyramid_top_k": "3"})
        self.assertTrue(tm.RUN_CFG["PYRAMID_MULTI_CANDIDATE"])
        self.assertEqual(tm.RUN_CFG["PYRAMID_TOP_K"], 3)
        self.assertEqual(tm._pyramid_topk_settings()[:2], (True, 3))

    def test_missing_keys_leave_run_cfg_untouched(self):
        # headless/TM_RUN_CFG_JSON ile verilen degerler UI yokken ezilmemeli.
        tm.RUN_CFG["PYRAMID_MULTI_CANDIDATE"] = True
        tm.RUN_CFG["PYRAMID_TOP_K"] = 5
        tm.apply_pyramid_ui_state({"kalman": False})
        self.assertTrue(tm.RUN_CFG["PYRAMID_MULTI_CANDIDATE"])
        self.assertEqual(tm.RUN_CFG["PYRAMID_TOP_K"], 5)

    def test_invalid_k_is_ignored(self):
        tm.RUN_CFG["PYRAMID_TOP_K"] = 5
        for bad in ("", "abc", None, "0", "-2"):
            tm.apply_pyramid_ui_state({"pyramid_top_k": bad})
            self.assertEqual(tm.RUN_CFG["PYRAMID_TOP_K"], 5, bad)

    def test_k_one_disables_multi_candidate_path(self):
        tm.apply_pyramid_ui_state({"multi_candidate": True, "pyramid_top_k": "1"})
        self.assertFalse(tm._pyramid_topk_settings()[0])

    def test_next_pyramid_top_k_cycles(self):
        self.assertEqual(tm.next_pyramid_top_k("1"), "3")
        self.assertEqual(tm.next_pyramid_top_k("3"), "5")
        self.assertEqual(tm.next_pyramid_top_k("5"), "1")
        self.assertEqual(tm.next_pyramid_top_k(None), "1")  # bilinmeyen -> bastan

    def test_runtime_panel_exposes_both_controls(self):
        keys = {b["key"] for b in tm._build_runtime_buttons()}
        self.assertIn("multi_candidate", keys)
        self.assertIn("pyramid_top_k", keys)

    def test_cycle_click_dispatches_per_key(self):
        import cv2
        buttons = tm._build_runtime_buttons()
        state = {"model_norm": tm.MODEL_INPUT_NORM, "pyramid_top_k": "1",
                 "multi_candidate": False, "_panel_collapsed": False, "_hover_key": None}
        img = np.zeros((960, 1280, 3), dtype=np.uint8)
        tm._draw_runtime_buttons(img, state, buttons, display_size=(1280, 960))
        button = next(b for b in buttons if b["key"] == "pyramid_top_k")
        bx, by, bw, bh = button["rect"]
        tm._runtime_buttons_mouse_cb(
            cv2.EVENT_LBUTTONDOWN, bx + bw // 2, by + bh // 2, 0,
            {"state": state, "buttons": buttons},
        )
        # K butonu K'yi dondurmeli, on isleme modunu DEGISTIRMEMELI.
        self.assertEqual(state["pyramid_top_k"], "3")
        self.assertEqual(state["model_norm"], tm.MODEL_INPUT_NORM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
