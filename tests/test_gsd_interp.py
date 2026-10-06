# -*- coding: utf-8 -*-
"""GSD_RESIZE_INTERP ve BENCHMARK arama-merkezi korumasi icin bagimliliksiz testler.

Ana betik cv2/osgeo/tensorflow gerektirdigi icin burada import edilmez; kaynak
AST ile okunur. Amac:
  1) Varsayilanlarin onceki davranisi korudugunu garanti etmek
     (GSD_RESIZE_INTERP = "NEAREST" -> sonuclar birebir ayni).
  2) Yeni bayraklarin dogru yerlere baglandigini dogrulamak.

Calistirma:
    python -m unittest tests.test_gsd_interp -v
"""

import ast
import os
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_HERE)
_CORE_PY = os.path.join(
    _PROJECT_DIR,
    "template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py",
)

with open(_CORE_PY, encoding="utf-8") as _fh:
    _SOURCE = _fh.read()
_TREE = ast.parse(_SOURCE)


def _run_cfg_value(key):
    for node in _TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "RUN_CFG" for t in node.targets
        ) and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == key:
                    return ast.literal_eval(v)
    raise KeyError(key)


def _load_interp_function(mode):
    """_gsd_resize_interpolation'i sahte cv2 ile ayri bir ad alaninda calistirir."""
    fn = next(
        n for n in _TREE.body
        if isinstance(n, ast.FunctionDef) and n.name == "_gsd_resize_interpolation"
    )
    fake_cv2 = types.SimpleNamespace(INTER_NEAREST=0, INTER_LINEAR=1, INTER_CUBIC=2, INTER_AREA=3)
    ns = {
        "cv2": fake_cv2,
        "GSD_RESIZE_INTERP": mode,
        "MODEL_INPUT_INTERP_BY_NAME": {
            "NEAREST": 0, "AREA": 3, "LINEAR": 1, "CUBIC": 2,
        },
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), _CORE_PY, "exec"), ns)
    return ns["_gsd_resize_interpolation"], fake_cv2


class TestDefaultsPreserveOldBehavior(unittest.TestCase):
    def test_gsd_resize_default_is_nearest(self):
        self.assertEqual(_run_cfg_value("GSD_RESIZE_INTERP"), "NEAREST")

    def test_benchmark_fallback_default_off(self):
        self.assertIs(_run_cfg_value("BENCHMARK_ALLOW_SEARCH_CENTER_FALLBACK"), False)

    def test_nearest_mode_returns_inter_nearest(self):
        fn, cv2 = _load_interp_function("NEAREST")
        for scale in (0.05, 0.1, 0.5, 1.0, 2.0):
            self.assertEqual(fn(scale), cv2.INTER_NEAREST)


class TestInterpolationSelection(unittest.TestCase):
    def test_area_mode(self):
        fn, cv2 = _load_interp_function("AREA")
        self.assertEqual(fn(0.1), cv2.INTER_AREA)

    def test_auto_mode_downscale_area_upscale_linear(self):
        fn, cv2 = _load_interp_function("AUTO")
        self.assertEqual(fn(0.1), cv2.INTER_AREA)
        self.assertEqual(fn(1.5), cv2.INTER_LINEAR)


class TestWiring(unittest.TestCase):
    def test_all_gsd_resizes_use_flag(self):
        self.assertEqual(_SOURCE.count("interpolation=gsd_interp)"), 4)
        self.assertNotIn("cuda_resize_if_available(cr_image, (base_w, base_h), interpolation=cv2.INTER_NEAREST)", _SOURCE)

    def test_benchmark_guard_present(self):
        self.assertIn("if benchmark and not BENCHMARK_ALLOW_SEARCH_CENTER_FALLBACK:", _SOURCE)
        guard = _SOURCE.split("if benchmark and not BENCHMARK_ALLOW_SEARCH_CENTER_FALLBACK:", 1)[1][:800]
        self.assertIn("NO_INTERSECTION_USE_SEARCH_CENTER = False", guard)
        self.assertIn("LOW_SCORE_USE_SEARCH_CENTER = 0.0", guard)


if __name__ == "__main__":
    unittest.main()
