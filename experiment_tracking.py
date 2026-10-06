# -*- coding: utf-8 -*-
"""Yeniden uretilebilir calisma manifesti ve kare muhasebesi.

Bu modul agir goruntu/GIS kutuphanelerine bagli degildir. Ana lokalizasyon
betigi hata alsa bile o ana kadarki kare durumlari ayri bir CSV'ye yazilabilir.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional


FINAL_FRAME_STATUSES = frozenset({"accepted", "rejected_hold", "skipped", "failed"})


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_run_id(now: Optional[datetime] = None) -> str:
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str, hash_mode: str = "metadata") -> Dict[str, Any]:
    absolute = os.path.abspath(path)
    stat = os.stat(absolute)
    result: Dict[str, Any] = {
        "path": absolute,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if str(hash_mode).lower() == "sha256":
        result["sha256"] = sha256_file(absolute)
    return result


def environment_snapshot() -> Dict[str, Any]:
    package_names = (
        "numpy", "opencv-python", "tensorflow", "rasterio", "GDAL",
        "pyproj", "Pillow", "piexif", "affine", "pandas",
    )
    packages = {}
    for package_name in package_names:
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
    }


def git_revision(workdir: Optional[str] = None) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir or os.getcwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_is_dirty(workdir: Optional[str] = None) -> Optional[bool]:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workdir or os.getcwd(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _quantile(sorted_values: List[float], probability: float) -> Optional[float]:
    if not sorted_values:
        return None
    position = max(0.0, min(1.0, float(probability))) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _numeric_summary(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    clean = sorted(
        float(value) for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not clean:
        return {"count": 0, "p50": None, "p95": None, "max": None, "mae": None, "rmse": None}
    return {
        "count": len(clean),
        "p50": _quantile(clean, 0.50),
        "p95": _quantile(clean, 0.95),
        "max": max(clean),
        "mae": sum(abs(value) for value in clean) / len(clean),
        "rmse": math.sqrt(sum(value * value for value in clean) / len(clean)),
    }


@dataclass(frozen=True)
class FrameStatus:
    frame_index: int
    image_path: str
    status: str
    reason: str = ""
    mode: str = ""
    output_source: str = ""
    search_anchor_source: str = ""
    search_mode: str = ""
    altitude_source: str = ""
    motion_source: str = ""
    visual_fix_valid: Optional[bool] = None
    latitude_estimate: Optional[float] = None
    longitude_estimate: Optional[float] = None
    error_m: Optional[float] = None
    confidence: Optional[float] = None
    quality_reason: str = ""
    score_floor: Optional[float] = None
    peak_margin_floor: Optional[float] = None
    peak_sidelobe_ratio_floor: Optional[float] = None
    template_std_floor: Optional[float] = None
    intersection_mode: str = ""
    search_roi_size_px: Optional[int] = None
    next_search_roi_size_px: Optional[int] = None
    visualization_center_source: str = ""
    visualization_half_size_px: Optional[int] = None
    latency_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status not in FINAL_FRAME_STATUSES:
            raise ValueError(f"Gecersiz kare durumu: {self.status}")


class FrameStatusRecorder:
    """Her giris karesi icin tam olarak bir terminal durum saklar."""

    def __init__(self) -> None:
        self._rows: Dict[int, FrameStatus] = {}

    def record(self, row: FrameStatus) -> None:
        if row.frame_index in self._rows:
            raise ValueError(f"Kare zaten kaydedildi: {row.frame_index}")
        self._rows[row.frame_index] = row

    @property
    def rows(self) -> List[FrameStatus]:
        return [self._rows[index] for index in sorted(self._rows)]

    def summary(self, attempted_count: Optional[int] = None) -> Dict[str, Any]:
        counts = Counter(row.status for row in self.rows)
        attempted = int(attempted_count if attempted_count is not None else len(self._rows))
        accounted = len(self._rows)
        rejected_streak = 0
        max_rejected_streak = 0
        recovery_lengths = []
        for row in self.rows:
            if row.status == "rejected_hold":
                rejected_streak += 1
                max_rejected_streak = max(max_rejected_streak, rejected_streak)
            elif row.status == "accepted":
                if rejected_streak:
                    recovery_lengths.append(rejected_streak)
                rejected_streak = 0
            else:
                rejected_streak = 0
        accepted_rows = [row for row in self.rows if row.status == "accepted"]
        evaluable_rows = [row for row in self.rows if row.error_m is not None]
        return {
            "attempted": attempted,
            "accounted": accounted,
            "missing": max(0, attempted - accounted),
            "coverage": (counts.get("accepted", 0) / attempted) if attempted else 0.0,
            "counts": dict(sorted(counts.items())),
            "accepted_error_m": _numeric_summary(row.error_m for row in accepted_rows),
            "all_output_error_m": _numeric_summary(row.error_m for row in evaluable_rows),
            "latency_ms": _numeric_summary(row.latency_ms for row in self.rows),
            "dropout": {
                "max_consecutive_rejected_hold": max_rejected_streak,
                "open_rejected_hold_at_end": rejected_streak,
                "recovery_event_count": len(recovery_lengths),
                "recovery_frames_p50": _quantile(sorted(float(v) for v in recovery_lengths), 0.50),
                "recovery_frames_max": max(recovery_lengths) if recovery_lengths else None,
            },
        }

    def write_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fieldnames = list(FrameStatus.__dataclass_fields__.keys())
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(asdict(row))
        os.replace(temporary_path, path)


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)


def build_run_manifest(
    *,
    run_id: str,
    mode: str,
    resolved_config: Mapping[str, Any],
    assets: Iterable[str],
    hash_mode: str = "metadata",
    frame_summary: Optional[Mapping[str, Any]] = None,
    asset_fingerprints: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    if asset_fingerprints is None:
        asset_rows = []
        for path in assets:
            if path and os.path.isfile(path):
                asset_rows.append(file_fingerprint(path, hash_mode=hash_mode))
    else:
        asset_rows = [dict(row) for row in asset_fingerprints]
    return {
        "schema_version": 1,
        "run_id": str(run_id),
        "created_utc": utc_timestamp(),
        "mode": str(mode),
        "resolved_config": dict(resolved_config),
        "asset_hash_mode": str(hash_mode).lower(),
        "assets": asset_rows,
        "environment": environment_snapshot(),
        "source_revision": git_revision(os.path.dirname(os.path.abspath(__file__))),
        "source_dirty": git_is_dirty(os.path.dirname(os.path.abspath(__file__))),
        "frame_summary": dict(frame_summary or {}),
    }
