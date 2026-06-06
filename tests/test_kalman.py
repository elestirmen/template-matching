# -*- coding: utf-8 -*-
"""
PositionKalmanFilter (sabit-konum 2B Kalman filtresi) birim testleri.

Tasarim: simulasyon projesindeki PositionKalmanFilter ile ayni (per-eksen skaler,
hiz durumu yok, confidence-olcekli olcum gurultusu). Hiz ekstrapole etmedigi icin
sapamaz; her gecerli olcumde olcume dogru cekilir.

Bu testler PRODUKSIYON kodunu DEGISTIRMEZ; ana betigi (monolit) bir modul gibi
yukleyip yalnizca PositionKalmanFilter sinifini dogrular.

Calistirma:
    python -m unittest tests.test_kalman -v
    python -m unittest discover -s tests -v

Not: Ana modulun yuklenmesi cv2 / osgeo / tensorflow gerektirir; biri yoksa testler
otomatik ATLANIR (skip).
"""

import os
import importlib.util
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_HERE)
_CORE_PY = os.path.join(
    _PROJECT_DIR,
    "template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py",
)

tm = None
_IMPORT_ERROR = None
try:
    import numpy as np  # noqa: F401  (diger testlerde kullanilabilir)
    _spec = importlib.util.spec_from_file_location("tm_core_under_test_kalman", _CORE_PY)
    tm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tm)
except Exception as exc:  # pragma: no cover - ortam bagimliligi
    _IMPORT_ERROR = exc


@unittest.skipIf(tm is None, f"Ana modul yuklenemedi (bagimlilik eksik olabilir): {_IMPORT_ERROR}")
class TestPositionKalmanFilter(unittest.TestCase):
    """Sabit-konum modeli: predict (belirsizlik buyutme), confidence-olcekli update."""

    def _make(self, x0=100.0, y0=200.0, process_noise=50.0, measurement_noise=80.0):
        return tm.PositionKalmanFilter((x0, y0), process_noise, measurement_noise)

    def test_init_position(self):
        kf = self._make(10.0, 20.0)
        self.assertEqual(kf.position, (10, 20))

    def test_position_returns_ints(self):
        kf = self._make(10.4, 20.6)
        px, py = kf.position
        self.assertIsInstance(px, int)
        self.assertIsInstance(py, int)
        self.assertEqual((px, py), (10, 21))  # yuvarlama

    def test_predict_with_motion_shifts_position(self):
        kf = self._make(100.0, 200.0)
        kf.predict(15.0, -25.0)
        self.assertEqual(kf.position, (115, 175))

    def test_predict_zero_motion_keeps_position(self):
        kf = self._make(100.0, 200.0)
        kf.predict(0.0, 0.0)
        self.assertEqual(kf.position, (100, 200))

    def test_predict_inflates_uncertainty(self):
        kf = self._make(process_noise=50.0)
        u_before = kf.uncertainty_px
        kf.predict(0.0, 0.0)
        self.assertGreater(kf.uncertainty_px, u_before)

    def test_update_pulls_toward_measurement(self):
        kf = self._make(0.0, 0.0)
        kf.update(100.0, 0.0, confidence=1.0)
        px, _ = kf.position
        # Olcume dogru kayar ama (tek adimda) tam ustune oturmaz.
        self.assertGreater(px, 0)
        self.assertLess(px, 100)

    def test_update_reduces_uncertainty(self):
        kf = self._make()
        u_before = kf.uncertainty_px
        kf.update(120.0, 220.0, confidence=1.0)
        self.assertLess(kf.uncertainty_px, u_before)

    def test_higher_confidence_pulls_more(self):
        # Ayni baslangic/olcum: yuksek confidence olcume daha cok yaklasmali.
        lo = self._make(0.0, 0.0); lo.predict(0.0, 0.0)
        hi = self._make(0.0, 0.0); hi.predict(0.0, 0.0)
        lo.update(100.0, 0.0, confidence=0.2)
        hi.update(100.0, 0.0, confidence=1.0)
        self.assertGreater(hi.position[0], lo.position[0])

    def test_converges_to_stationary_measurement(self):
        # Sabit bir olcume tekrar tekrar guncellenince konum o olcume yakinsamali.
        kf = self._make(0.0, 0.0)
        for _ in range(60):
            kf.predict(0.0, 0.0)
            kf.update(300.0, -150.0, confidence=1.0)
        px, py = kf.position
        self.assertAlmostEqual(px, 300, delta=5)
        self.assertAlmostEqual(py, -150, delta=5)

    def test_smooths_noisy_measurements(self):
        # Sabit gercek konum + +/-40px gurultu: filtre ciktisi olcumlerden
        # belirgin sekilde daha az sapmali (yumusatma).
        true_x, true_y = 500.0, 500.0
        noise = [40, -40, 35, -38, 42, -36, 39, -41, 37, -39] * 3
        kf = self._make(true_x, true_y)
        filt_sqerr = 0.0
        meas_sqerr = 0.0
        for n in noise:
            mx, my = true_x + n, true_y - n
            kf.predict(0.0, 0.0)
            kf.update(mx, my, confidence=1.0)
            px, py = kf.position
            filt_sqerr += (px - true_x) ** 2 + (py - true_y) ** 2
            meas_sqerr += (mx - true_x) ** 2 + (my - true_y) ** 2
        self.assertLess(filt_sqerr, meas_sqerr)

    def test_reset_reseeds_to_measurement(self):
        # reset() konumu olcume tasir ve belirsizligi olcum-gurultusu seviyesine ceker.
        kf = self._make(0.0, 0.0, measurement_noise=80.0)
        for _ in range(5):
            kf.predict(0.0, 0.0)  # belirsizlik buyusun
        u_inflated = kf.uncertainty_px
        kf.reset((1234.0, 5678.0))
        self.assertEqual(kf.position, (1234, 5678))
        self.assertLess(kf.uncertainty_px, u_inflated)
        self.assertAlmostEqual(kf.uncertainty_px, 80.0, delta=1e-6)

    def test_coast_when_no_update(self):
        # Update yapilmazsa konum sabit kalir (predict(0,0) sadece belirsizligi buyutur).
        kf = self._make(123.0, 456.0)
        for _ in range(5):
            kf.predict(0.0, 0.0)  # update yok -> coast
        self.assertEqual(kf.position, (123, 456))

    def test_predict_process_var_override(self):
        # predict(process_var=...) bu kareye ozel q kullanir (covaryans modu).
        kf = self._make(0.0, 0.0, process_noise=10.0, measurement_noise=10.0)
        v0 = kf.position_var_mean
        kf.predict(0.0, 0.0, process_var=40000.0)  # std 200 -> var +40000
        self.assertAlmostEqual(kf.position_var_mean, v0 + 40000.0, delta=1e-6)

    def test_gain_max_caps_movement(self):
        # gain_max=0.5 -> olcume EN FAZLA %50 cekilir (tavansiz ~tam cekerdi).
        kf = self._make(0.0, 0.0, process_noise=1000.0, measurement_noise=1.0)
        kf.predict(0.0, 0.0)  # var cok buyur -> tavansiz kazanc ~1.0
        kf.update(100.0, 0.0, confidence=1.0, gain_max=0.5)
        self.assertAlmostEqual(kf.position[0], 50, delta=2)  # ~%50

    def test_gain_max_one_is_uncapped(self):
        # gain_max=1.0 -> mevcut davranis (yuksek var'da olcume neredeyse tam oturur).
        kf = self._make(0.0, 0.0, process_noise=1000.0, measurement_noise=1.0)
        kf.predict(0.0, 0.0)
        kf.update(100.0, 0.0, confidence=1.0, gain_max=1.0)
        self.assertGreater(kf.position[0], 95)

    def test_lower_gain_max_smooths_more(self):
        # Dusuk tavan -> olcume daha az cekilir (daha cok yumusatma).
        a = self._make(0.0, 0.0, process_noise=1000.0, measurement_noise=1.0); a.predict(0.0, 0.0)
        b = self._make(0.0, 0.0, process_noise=1000.0, measurement_noise=1.0); b.predict(0.0, 0.0)
        a.update(100.0, 0.0, confidence=1.0, gain_max=0.3)
        b.update(100.0, 0.0, confidence=1.0, gain_max=0.8)
        self.assertLess(a.position[0], b.position[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
