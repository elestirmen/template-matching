# -*- coding: utf-8 -*-
"""Iki deney modunu acik ve yeniden uretilebilir bicimde calistiran CLI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


MAIN_PATTERN = "template_matching_parallel*.py"


def build_config(
    *,
    mode: str,
    max_frames: int = 0,
    sha256_assets: bool = False,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = dict(extra_config or {})
    if mode == "benchmark":
        config.update({
            "BENCHMARK": True,
            "STRICT_GPS_DENIED_INFERENCE": False,
            "USE_EXIF_MOTION_SEARCH_PRIOR": False,
            "USE_GPS_REVERT": False,
            "ALTITUDE_SOURCE": "exif_altitude_proxy",
        })
    elif mode == "simulation":
        config.update({
            "BENCHMARK": False,
            "STRICT_GPS_DENIED_INFERENCE": True,
            "USE_EXIF_MOTION_SEARCH_PRIOR": False,
            "USE_GPS_REVERT": False,
            # Veri setindeki hazir/simule irtifa her karede kullanilir; yatay GPS
            # konumu t=0 sonrasinda yine localizer'dan izole edilir.
            "ALTITUDE_SOURCE": "exif_altitude_proxy",
        })
    else:
        raise ValueError(f"Bilinmeyen mod: {mode}")
    config["MAX_FRAMES"] = max(0, int(max_frames))
    config["ASSET_HASH_MODE"] = "sha256" if sha256_assets else "metadata"
    config["WRITE_RUN_ARTIFACTS"] = True
    return config


def _load_pipeline(project_dir: Path):
    candidates = sorted(project_dir.glob(MAIN_PATTERN))
    if len(candidates) != 1:
        raise RuntimeError(f"Ana pipeline dosyasi tekil bulunamadi: {candidates}")
    spec = importlib.util.spec_from_file_location("template_matching_pipeline", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("Ana pipeline modulu yuklenemedi")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "simulation"), required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sha256-assets", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--config-json",
        default="{}",
        help="RUN_CFG icin ek JSON; mod guvenlik kurallari en son uygulanir.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    extra = json.loads(args.config_json)
    if not isinstance(extra, dict):
        raise ValueError("--config-json bir JSON nesnesi olmali")
    config = build_config(
        mode=args.mode,
        max_frames=args.max_frames,
        sha256_assets=args.sha256_assets,
        extra_config=extra,
    )
    os.environ["TM_RUN_CFG_JSON"] = json.dumps(config, ensure_ascii=True)
    module = _load_pipeline(Path(__file__).resolve().parent)
    module.setup_logging()
    callbacks = {"on_frame": lambda *_args: None} if args.headless else None
    module.run_pipeline(callbacks=callbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
