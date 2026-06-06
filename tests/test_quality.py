# -*- coding: utf-8 -*-
"""gps_denied_autonomy modulu (lokalizasyon kalitesi + sensor fuzyonu) birim testleri.

Bu modul SAF Python'dur (cv2/osgeo/tensorflow GEREKTIRMEZ), bu yuzden testler
herhangi bir Python'da dogrudan calisir. Uretim kodunu degistirmez.

Calistirma:
    python -m unittest tests.test_quality -v
    python -m unittest discover -s tests -v
"""

import os
import sys
import math
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_HERE)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import gps_denied_autonomy as gda  # noqa: E402


def _box(cx, cy, size=512):
    """Merkezi (cx,cy) olan (x,y,w,h) kutusu uretir."""
    return (int(cx - size // 2), int(cy - size // 2), size, size)


class TestNormalizeMatchScore(unittest.TestCase):
    def test_ccoeff_range_maps_to_unit(self):
        # CCOEFF_NORMED [-1,1] -> [0,1]
        self.assertAlmostEqual(gda.normalize_match_score(1.0, False), 1.0)
        self.assertAlmostEqual(gda.normalize_match_score(-1.0, False), 0.0)
        self.assertAlmostEqual(gda.normalize_match_score(0.0, False), 0.5)
        self.assertAlmostEqual(gda.normalize_match_score(0.2, False), 0.6)

    def test_sqdiff_inverts(self):
        # SQDIFF: dusuk = iyi -> tersine cevrilir
        self.assertAlmostEqual(gda.normalize_match_score(0.0, True), 1.0)
        self.assertAlmostEqual(gda.normalize_match_score(1.0, True), 0.0)
        self.assertAlmostEqual(gda.normalize_match_score(0.25, True), 0.75)

    def test_output_always_in_unit_interval(self):
        for v in (-5.0, -1.0, 0.0, 0.3, 1.0, 100.0):
            for sq in (True, False):
                out = gda.normalize_match_score(v, sq)
                self.assertGreaterEqual(out, 0.0)
                self.assertLessEqual(out, 1.0)


class TestComputeLocalizationQuality(unittest.TestCase):
    def setUp(self):
        self.score_thr = 0.35
        self.conf_thr = 0.40
        self.spread_thr = 120.0

    def _quality(self, scores, boxes, pred_box, mode):
        return gda.compute_localization_quality(
            scores, boxes, pred_box, mode, False,
            self.score_thr, self.conf_thr, self.spread_thr,
        )

    def test_good_triplet_is_reliable(self):
        # Yuksek skor + uc kutu kesisim merkezine yakin (dusuk yayilim) -> guvenilir.
        center = (1000, 1000)
        boxes = [_box(995, 1002), _box(1003, 998), _box(1000, 1001)]
        pred = _box(1000, 1000)
        q = self._quality([0.6, 0.62, 0.61], boxes, pred, "abc")
        self.assertTrue(q.is_reliable)
        self.assertEqual(q.reason, "ok")
        self.assertGreater(q.confidence, self.conf_thr)
        self.assertLess(q.center_spread_px, self.spread_thr)

    def test_low_score_floor_unreliable(self):
        # Bir sablon cok dusuk skor -> skor tabani esigin altinda -> guvenilmez.
        center = (1000, 1000)
        boxes = [_box(1000, 1000), _box(1000, 1000), _box(1000, 1000)]
        pred = _box(1000, 1000)
        # -0.5 -> normalize 0.25 < 0.35
        q = self._quality([0.6, 0.6, -0.5], boxes, pred, "abc")
        self.assertFalse(q.is_reliable)
        self.assertEqual(q.reason, "score_floor")

    def test_high_spread_unreliable(self):
        # Skorlar iyi ama kutular dagilmis (buyuk yayilim) -> geometrik tutarsiz.
        pred = _box(1000, 1000)
        boxes = [_box(1000, 1000), _box(1400, 1000), _box(1000, 1400)]
        q = self._quality([0.7, 0.7, 0.7], boxes, pred, "abc")
        self.assertFalse(q.is_reliable)
        self.assertEqual(q.reason, "spread")
        self.assertGreater(q.center_spread_px, self.spread_thr)

    def test_center_fallback_low_confidence(self):
        # Kesisim yok (center_fallback) + orta skor -> dusuk guven -> guvenilmez.
        pred = _box(1000, 1000)
        boxes = [_box(1000, 1000), _box(1000, 1000), _box(1000, 1000)]
        q = self._quality([0.4, 0.4, 0.4], boxes, pred, "center_fallback")
        # score_floor 0.7 >= 0.35, spread 0 -> reddedilmezse confidence esigi devreye girer
        self.assertIn(q.reason, ("confidence", "ok"))

    def test_no_boxes_infinite_spread(self):
        pred = _box(1000, 1000)
        q = self._quality([0.6, 0.6, 0.6], [], pred, "abc")
        self.assertFalse(math.isfinite(q.center_spread_px))
        self.assertFalse(q.is_reliable)


class TestFuseMeasurementWithPrior(unittest.TestCase):
    def _q(self, reliable=True, confidence=1.0):
        return gda.LocalizationQuality(
            normalized_scores=(0.6, 0.6, 0.6), score_floor=0.6, score_mean=0.6,
            center_spread_px=5.0, confidence=confidence, is_reliable=reliable, reason="ok",
        )

    def test_no_prior_returns_measurement(self):
        fused, ok, jump = gda.fuse_measurement_with_prior(
            None, (1000, 1000), self._q(), 600.0, 0.75)
        self.assertEqual(fused, (1000, 1000))
        self.assertEqual(jump, 0.0)

    def test_far_jump_rejected(self):
        prior = (1000, 1000)
        meas = (5000, 5000)  # >> 600*1.75
        fused, ok, jump = gda.fuse_measurement_with_prior(prior, meas, self._q(), 600.0, 0.75)
        self.assertEqual(fused, prior)
        self.assertFalse(ok)
        self.assertGreater(jump, 600.0 * 1.75)

    def test_unreliable_rejected(self):
        prior = (1000, 1000)
        meas = (1050, 1000)
        fused, ok, jump = gda.fuse_measurement_with_prior(
            prior, meas, self._q(reliable=False), 600.0, 0.75)
        self.assertEqual(fused, prior)
        self.assertFalse(ok)

    def test_blend_between_prior_and_measurement(self):
        prior = (1000, 1000)
        meas = (1100, 1000)
        # gain 0.5 * confidence 1.0 -> efektif 0.5 -> orta nokta
        fused, ok, jump = gda.fuse_measurement_with_prior(
            prior, meas, self._q(confidence=1.0), 600.0, 0.5)
        self.assertTrue(ok)
        self.assertEqual(fused[0], 1050)
        self.assertEqual(fused[1], 1000)

    def test_confidence_scales_gain(self):
        prior = (0, 0)
        meas = (100, 0)
        # gain 1.0 * confidence 0.25 -> efektif 0.25 -> x=25
        fused, ok, _ = gda.fuse_measurement_with_prior(
            prior, meas, self._q(confidence=0.25), 600.0, 1.0)
        self.assertEqual(fused[0], 25)


class TestPlanKalmanMeasurementAction(unittest.TestCase):
    """Fiziksel hareket kapisi + yeniden-kazanim karar fonksiyonu."""

    REACQ_JUMP = 300.0
    REACQ_FRAMES = 2

    def _plan(self, reliable, three_way, dist, max_step, reacq_count):
        return gda.plan_kalman_measurement_action(
            reliable, three_way, dist, max_step,
            self.REACQ_JUMP, reacq_count, self.REACQ_FRAMES,
        )

    # --- max_step=0: mevcut davranis BIREBIR korunmali ---
    def test_disabled_reliable_near_updates(self):
        self.assertEqual(self._plan(True, True, 50, 0, 0), ("update", 0))
        self.assertEqual(self._plan(True, False, 250, 0, 0), ("update", 0))  # 2'li uzak da update (eski)

    def test_disabled_unreliable_coasts(self):
        self.assertEqual(self._plan(False, True, 10, 0, 0), ("coast_unreliable", 0))

    def test_disabled_far_three_way_reacquires(self):
        # 3'lu + REACQUIRE_JUMP asildi -> streak; ikinci teyitte re-seed
        self.assertEqual(self._plan(True, True, 400, 0, 0), ("reacq_wait", 1))
        self.assertEqual(self._plan(True, True, 400, 0, 1), ("reseed", 0))

    def test_disabled_far_two_way_still_updates(self):
        # 2'li uzak (REACQUIRE_JUMP ustu) eski mantikta yine update (bilinen zayiflik)
        self.assertEqual(self._plan(True, False, 1000, 0, 0), ("update", 0))

    # --- max_step>0: fiziksel kapi devrede ---
    def test_gate_rejects_far_two_way(self):
        # 2'li uzak olcum (kapinin ustunde) -> coast_outlier (ISINLANMA YOK)
        self.assertEqual(self._plan(True, False, 1000, 250, 0), ("coast_outlier", 0))
        self.assertEqual(self._plan(True, False, 300, 250, 0), ("coast_outlier", 0))

    def test_gate_allows_plausible_step(self):
        self.assertEqual(self._plan(True, False, 200, 250, 0), ("update", 0))
        self.assertEqual(self._plan(True, True, 249, 250, 0), ("update", 0))

    def test_gate_far_three_way_needs_streak(self):
        # max_step (250) asan 3'lu -> tek karede update YOK; streak sonra re-seed
        self.assertEqual(self._plan(True, True, 260, 250, 0), ("reacq_wait", 1))
        self.assertEqual(self._plan(True, True, 260, 250, 1), ("reseed", 0))

    def test_gate_unreliable_still_coasts(self):
        self.assertEqual(self._plan(False, True, 1000, 250, 5), ("coast_unreliable", 0))


class TestEffectiveStepGate(unittest.TestCase):
    """Adaptif hareket kapisi esigi (medyan adimin kati)."""

    def test_multiplier_zero_returns_floor(self):
        # Adaptif kapali -> floor (mutlak ust sinir / 0=kapali) dondurur
        self.assertEqual(gda.effective_step_gate_px([10, 20, 30], 0.0, 250.0), 250.0)
        self.assertEqual(gda.effective_step_gate_px([10, 20, 30], 0.0, 0.0), 0.0)

    def test_warmup_no_gate(self):
        # Yeterli ornek yok (min_samples=3) -> 0 (gateleme yok)
        self.assertEqual(gda.effective_step_gate_px([], 4.0, 250.0), 0.0)
        self.assertEqual(gda.effective_step_gate_px([100, 100], 4.0, 250.0), 0.0)

    def test_adaptive_scales_with_median(self):
        # medyan 300 -> 4*300 = 1200 (floor 250'nin ustunde)
        steps = [280, 300, 320, 310, 290]
        self.assertEqual(gda.effective_step_gate_px(steps, 4.0, 250.0), 1200.0)

    def test_floor_applies_for_slow_motion(self):
        # Yavas hareket: medyan ~20 -> 4*20=80 < floor -> floor (250) doner
        self.assertEqual(gda.effective_step_gate_px([18, 20, 22, 19], 4.0, 250.0), 250.0)

    def test_median_robust_to_outlier(self):
        # Tek buyuk sicrama medyani bozmaz: medyan 100 -> 400
        steps = [90, 100, 110, 100, 5000]  # medyan 100
        self.assertEqual(gda.effective_step_gate_px(steps, 4.0, 100.0), 400.0)

    def test_genuine_motion_never_frozen(self):
        # Tipik adim ~ medyan; efektif kapi = 4*medyan > medyan -> gercek adim HEP gecer
        steps = [400] * 6
        gate = gda.effective_step_gate_px(steps, 4.0, 250.0)
        self.assertGreater(gate, 400.0)  # 1600 > 400 -> donmaz


class TestBlendVelocityEma(unittest.TestCase):
    """Hareket ongorusu: hiz EMA guncellemesi."""

    def test_gain_one_takes_displacement(self):
        self.assertEqual(gda.blend_velocity_ema((10.0, 5.0), (40.0, -20.0), 1.0), (40.0, -20.0))

    def test_gain_zero_keeps_prev(self):
        self.assertEqual(gda.blend_velocity_ema((10.0, 5.0), (40.0, -20.0), 0.0), (10.0, 5.0))

    def test_half_blend(self):
        vx, vy = gda.blend_velocity_ema((10.0, 0.0), (30.0, 10.0), 0.5)
        self.assertAlmostEqual(vx, 20.0)
        self.assertAlmostEqual(vy, 5.0)

    def test_gain_clamped(self):
        # gain>1 kirpildigi icin displacement'i gecmez (1.0 gibi davranir)
        self.assertEqual(gda.blend_velocity_ema((10.0, 5.0), (40.0, -20.0), 5.0), (40.0, -20.0))


class TestCovarianceHelpers(unittest.TestCase):
    """Covaryans-tabanli kapi/ROI + medyan yardimcilari."""

    def test_median_basic(self):
        self.assertEqual(gda.median_of([]), 0.0)
        self.assertEqual(gda.median_of([5]), 5.0)
        self.assertEqual(gda.median_of([3, 1, 2]), 2.0)
        self.assertEqual(gda.median_of([4, 1, 3, 2]), 2.5)

    def test_innovation_gate_scales_with_uncertainty(self):
        # esik = sigma*sqrt(P+R)
        self.assertAlmostEqual(gda.innovation_gate_px(0.0, 0.0, 3.0), 0.0)
        self.assertAlmostEqual(gda.innovation_gate_px(16.0, 0.0, 3.0), 12.0)   # 3*sqrt(16)
        self.assertAlmostEqual(gda.innovation_gate_px(9.0, 16.0, 2.0), 10.0)   # 2*sqrt(25)

    def test_innovation_gate_grows_when_P_grows(self):
        # Coast'ta P buyur -> kapi genisler (kurtarma)
        small = gda.innovation_gate_px(100.0, 64.0, 3.0)
        big = gda.innovation_gate_px(10000.0, 64.0, 3.0)
        self.assertGreater(big, small)

    def test_window_size_clamped_to_base_and_max(self):
        # Cok kucuk belirsizlik -> taban; cok buyuk -> max
        base, mx, tpl = 2048.0, 8000.0, 512.0
        self.assertEqual(gda.covariance_window_size_px(1.0, 1.0, 4.0, tpl, base, mx), base)
        self.assertEqual(gda.covariance_window_size_px(1e9, 0.0, 4.0, tpl, base, mx), mx)

    def test_window_size_scales_between(self):
        # 2*sigma*sqrt(P+R)+tpl, taban/max arasinda
        # P=10000,R=0,sigma=4 -> 2*4*100 + 512 = 1312; taban 800 -> 1312
        val = gda.covariance_window_size_px(10000.0, 0.0, 4.0, 512.0, 800.0, 8000.0)
        self.assertAlmostEqual(val, 1312.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
