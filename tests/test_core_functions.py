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


if __name__ == "__main__":
    unittest.main(verbosity=2)
