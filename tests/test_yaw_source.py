# -*- coding: utf-8 -*-
"""YAW_SOURCE (ucus / gimbal yawi) icin bagimliliksiz testler.

Ana betik import edilmez (cv2/osgeo/tensorflow); kaynak AST ile okunur.
  1) Varsayilan "FLIGHT" = onceki davranis.
  2) "GIMBAL" iken rota profillerindeki YAW_OFFSET_DEG uygulanmaz (cift duzeltme
     olmasin), diger profil anahtarlari uygulanmaya devam eder.

Calistirma:
    python -m unittest tests.test_yaw_source -v
"""

import ast
import os
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE_PY = os.path.join(
    os.path.dirname(_HERE),
    "template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py",
)
with open(_CORE_PY, encoding="utf-8") as _fh:
    _SOURCE = _fh.read()
_TREE = ast.parse(_SOURCE)


def _module_dict(name):
    for node in _TREE.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise KeyError(name)


def _run_cfg_value(key):
    d = _module_dict("RUN_CFG")
    for k, v in zip(d.keys, d.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return ast.literal_eval(v)
    raise KeyError(key)


def _apply_profiles(run_cfg, route, explicit=()):
    fn = next(n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_route_profile_overrides")
    ns = {
        "RUN_CFG": run_cfg,
        "ROUTE_PROFILE_OVERRIDES": ast.literal_eval(_module_dict("ROUTE_PROFILE_OVERRIDES")),
        "_route_name_from_cfg": lambda: route,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), _CORE_PY, "exec"), ns)
    ns["_apply_route_profile_overrides"](set(explicit))
    return run_cfg


class TestYawSource(unittest.TestCase):
    def test_default_is_flight(self):
        self.assertEqual(_run_cfg_value("YAW_SOURCE"), "FLIGHT")

    def test_flight_applies_route_yaw_offset(self):
        cfg = _apply_profiles({"USE_ROUTE_PROFILES": True, "YAW_SOURCE": "FLIGHT", "YAW_OFFSET_DEG": 0.0}, "4_tezde_ucus_8")
        self.assertEqual(cfg["YAW_OFFSET_DEG"], -8.5)

    def test_gimbal_skips_route_yaw_offset(self):
        cfg = _apply_profiles({"USE_ROUTE_PROFILES": True, "YAW_SOURCE": "GIMBAL", "YAW_OFFSET_DEG": 0.0}, "4_tezde_ucus_8")
        self.assertEqual(cfg["YAW_OFFSET_DEG"], 0.0)

    def test_gimbal_keeps_other_profile_keys(self):
        cfg = _apply_profiles({"USE_ROUTE_PROFILES": True, "YAW_SOURCE": "GIMBAL", "KALMAN_WINDOW_FOLLOWS": True}, "3_tezde_ucus_7")
        self.assertIs(cfg["KALMAN_WINDOW_FOLLOWS"], False)

    def test_explicit_key_still_wins(self):
        cfg = _apply_profiles({"USE_ROUTE_PROFILES": True, "YAW_SOURCE": "FLIGHT", "YAW_OFFSET_DEG": 1.0},
                              "4_tezde_ucus_8", explicit=("YAW_OFFSET_DEG",))
        self.assertEqual(cfg["YAW_OFFSET_DEG"], 1.0)

    def test_gimbal_yaw_parsed_and_used(self):
        self.assertIn("GimbalDegree[^0-9+-]*", _SOURCE)
        self.assertIn("'gimbal_yaw': gimbal_yaw,", _SOURCE)
        self.assertIn('if YAW_SOURCE == "GIMBAL":', _SOURCE)


if __name__ == "__main__":
    unittest.main()
