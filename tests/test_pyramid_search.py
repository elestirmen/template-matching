# -*- coding: utf-8 -*-
"""pyramid_search modulu icin birim testleri (cok adayli kaba arama).

Calistirma:
    python -m unittest tests.test_pyramid_search -v

Not: Yalnizca numpy + opencv gerektirir; tensorflow/gdal yuklenmez.
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_HERE)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

ps = None
_IMPORT_ERROR = None
try:
    import numpy as np
    import cv2
    import pyramid_search as ps
except Exception as exc:  # pragma: no cover - ortam bagimliligi
    _IMPORT_ERROR = exc


@unittest.skipIf(ps is None, f"pyramid_search yuklenemedi: {_IMPORT_ERROR}")
class TestNmsTopCandidates(unittest.TestCase):
    """nms_top_candidates(): ayrik tepeler, bastirma yaricapi, sira."""

    def test_returns_separated_peaks_in_score_order(self):
        response = np.zeros((200, 200), dtype=np.float32)
        response[20, 30] = 0.9
        response[150, 160] = 0.8
        response[100, 40] = 0.7
        found = ps.nms_top_candidates(response, top_k=3, radius=10)
        self.assertEqual(
            [(c.x, c.y) for c in found],
            [(30, 20), (160, 150), (40, 100)],
        )
        self.assertAlmostEqual(found[0].score, 0.9, places=5)

    def test_suppresses_neighbours_of_same_peak(self):
        # Tek genis tepe: yaricap icindeki komsular ayri aday sayilmamali.
        # (Arka plan -1; duz bolge sahte aday uretmesin diye.)
        response = np.full((100, 100), -1.0, dtype=np.float32)
        response[50, 50] = 1.0
        response[50, 51] = 0.99
        response[51, 50] = 0.98
        response[10, 10] = 0.5  # uzaktaki ikinci gercek tepe
        found = ps.nms_top_candidates(response, top_k=3, radius=5)
        self.assertEqual([(c.x, c.y) for c in found][:2], [(50, 50), (10, 10)])
        # Ilk tepenin bastirma penceresinden ikinci bir aday cikmamali.
        self.assertFalse(
            any(abs(c.x - 50) <= 5 and abs(c.y - 50) <= 5 for c in found[1:])
        )

    def test_top_k_one_matches_minmaxloc(self):
        rng = np.random.default_rng(0)
        response = rng.random((300, 300)).astype(np.float32)
        found = ps.nms_top_candidates(response, top_k=1, radius=7)
        _, max_value, _, max_location = cv2.minMaxLoc(response)
        self.assertEqual((found[0].x, found[0].y), (int(max_location[0]), int(max_location[1])))
        self.assertAlmostEqual(found[0].score, float(max_value), places=5)

    def test_empty_response_returns_empty(self):
        self.assertEqual(ps.nms_top_candidates(np.empty((0, 0), np.float32), 5, 4), [])

    def test_crosses_block_boundaries(self):
        # Blok boyutu 256; tepeler farkli bloklara dusuyor.
        response = np.zeros((600, 600), dtype=np.float32)
        response[10, 10] = 0.95
        response[300, 300] = 0.90
        response[500, 100] = 0.85
        found = ps.nms_top_candidates(response, top_k=3, radius=20)
        self.assertEqual([(c.x, c.y) for c in found], [(10, 10), (300, 300), (100, 500)])


@unittest.skipIf(ps is None, f"pyramid_search yuklenemedi: {_IMPORT_ERROR}")
class TestRefineCandidates(unittest.TestCase):
    """refine_candidates(): pencere disi dolgu, ikinci adaydan kurtarma."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.scene = rng.integers(0, 256, size=(400, 400), dtype=np.uint8)
        self.template = self.scene[300:340, 250:290].copy()  # gercek eslesme (250, 300)
        self.truth = (250, 300)  # (x, y) response uzayinda

    def test_window_outside_candidates_is_filled(self):
        wrong = ps.Candidate(x=20, y=20, score=1.0)
        res = ps.refine_candidates(self.scene, self.template, [wrong], roi_pad_px=10)
        # Gercek konum hicbir adayin penceresinde degil -> dolgu degeri kalmali.
        self.assertLess(res[self.truth[1], self.truth[0]], -1e8)

    def test_second_candidate_recovers_true_match(self):
        # 1. aday yanlis (kaba asamanin kilitlendigi sahte tepe), 2. aday dogru.
        # Tek adayli yol burada kaybederdi; TOP-K ile dogru konum bulunur.
        candidates = [
            ps.Candidate(x=20, y=20, score=1.0),
            ps.Candidate(x=self.truth[0] + 6, y=self.truth[1] - 5, score=0.9),
        ]
        res = ps.refine_candidates(self.scene, self.template, candidates, roi_pad_px=15)
        _, max_value, _, max_location = cv2.minMaxLoc(res)
        self.assertEqual((int(max_location[0]), int(max_location[1])), self.truth)
        self.assertAlmostEqual(max_value, 1.0, places=4)

    def test_response_shape_matches_matchtemplate(self):
        res = ps.refine_candidates(
            self.scene,
            self.template,
            [ps.Candidate(x=self.truth[0], y=self.truth[1], score=1.0)],
            roi_pad_px=12,
        )
        expected = cv2.matchTemplate(self.scene, self.template, cv2.TM_CCOEFF_NORMED)
        self.assertEqual(res.shape, expected.shape)
        self.assertEqual(res.dtype, np.float32)

    def test_template_larger_than_image_returns_empty(self):
        big = np.zeros((500, 500), dtype=np.uint8)
        res = ps.refine_candidates(
            self.scene, big, [ps.Candidate(x=0, y=0, score=1.0)], roi_pad_px=5
        )
        self.assertEqual(res.size, 0)


@unittest.skipIf(ps is None, f"pyramid_search yuklenemedi: {_IMPORT_ERROR}")
class TestCoarseToFineTopk(unittest.TestCase):
    """coarse_to_fine_topk(): uctan uca davranis ve tek adayli yol ile uyum."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.scene = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
        self.truth = (180, 260)  # (x, y)
        self.template = self.scene[
            self.truth[1] : self.truth[1] + 64, self.truth[0] : self.truth[0] + 64
        ].copy()

    def test_finds_exact_match(self):
        res = ps.coarse_to_fine_topk(
            self.scene,
            self.template,
            coarse_scale=0.5,
            roi_pad_px=26,
            top_k=5,
        )
        _, max_value, _, max_location = cv2.minMaxLoc(res)
        self.assertEqual((int(max_location[0]), int(max_location[1])), self.truth)
        self.assertAlmostEqual(max_value, 1.0, places=4)

    def test_top_k_one_equals_single_candidate_path(self):
        # top_k=1 -> davranis eski tek adayli yol ile ayni tepeyi vermeli.
        res_one = ps.coarse_to_fine_topk(
            self.scene, self.template, coarse_scale=0.5, roi_pad_px=26, top_k=1
        )
        res_many = ps.coarse_to_fine_topk(
            self.scene, self.template, coarse_scale=0.5, roi_pad_px=26, top_k=5
        )
        self.assertEqual(res_one.shape, res_many.shape)
        for res in (res_one, res_many):
            _, _, _, loc = cv2.minMaxLoc(res)
            self.assertEqual((int(loc[0]), int(loc[1])), self.truth)

    def test_more_candidates_cover_more_area(self):
        # K buyudukce incelenen (dolgu olmayan) alan artmali.
        def covered(k):
            res = ps.coarse_to_fine_topk(
                self.scene, self.template, coarse_scale=0.5, roi_pad_px=20, top_k=k
            )
            return int(np.count_nonzero(res > -1e8))

        self.assertGreater(covered(4), covered(1))

    def test_degenerate_coarse_scale_falls_back_to_center(self):
        # Kaba olcekte template haritadan buyuk kalir -> merkez yedegi devreye girer.
        res = ps.coarse_to_fine_topk(
            self.scene, self.template, coarse_scale=0.01, roi_pad_px=20, top_k=5
        )
        self.assertEqual(res.shape, (449, 449))
        self.assertGreater(int(np.count_nonzero(res > -1e8)), 0)

    def test_template_larger_than_image_returns_empty(self):
        res = ps.coarse_to_fine_topk(
            np.zeros((32, 32), np.uint8),
            np.zeros((64, 64), np.uint8),
            coarse_scale=0.5,
            roi_pad_px=8,
            top_k=3,
        )
        self.assertEqual(res.size, 0)

    def test_auto_nms_radius(self):
        self.assertEqual(ps.auto_nms_radius((64, 64)), 32)
        self.assertEqual(ps.auto_nms_radius((4, 4)), 8)  # taban deger


if __name__ == "__main__":
    unittest.main()
