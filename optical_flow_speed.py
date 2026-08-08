"""Ardisik kamera karelerinden GPS-bagimsiz yatay hiz tahmini."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class OpticalFlowSpeed:
    speed_mps: float = 0.0
    displacement_px: float = 0.0
    dt_s: float = 0.0
    tracked_points: int = 0
    inlier_ratio: float = 0.0
    valid: bool = False
    reason: str = "warmup"


class OpticalFlowSpeedEstimator:
    """Sparse LK optical flow ile goruntu-merkezi otelemesini olcer.

    Benzerlik donusumunun (donme + olcek + oteleme) goruntu merkezindeki
    hareketi kullanmak, yaw ve irtifa degisiminin saf donme/olcek etkisini
    hizdan buyuk olcude ayirir. Sonuc, verilen kamera GSD'si ile m/s'ye cevrilir.
    """

    def __init__(
        self,
        *,
        max_dimension: int = 960,
        max_corners: int = 800,
        min_tracks: int = 20,
        min_inlier_ratio: float = 0.45,
        ransac_threshold_px: float = 2.5,
    ) -> None:
        self.max_dimension = max(160, int(max_dimension))
        self.max_corners = max(50, int(max_corners))
        self.min_tracks = max(6, int(min_tracks))
        self.min_inlier_ratio = float(np.clip(min_inlier_ratio, 0.0, 1.0))
        self.ransac_threshold_px = max(0.5, float(ransac_threshold_px))
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None
        self._prev_gsd_m_per_px: Optional[float] = None
        self._prev_scale = 1.0

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_timestamp = None
        self._prev_gsd_m_per_px = None
        self._prev_scale = 1.0

    def _prepare(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        if gray is None or gray.size == 0:
            raise ValueError("bos goruntu")
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        elif gray.ndim != 2:
            raise ValueError("goruntu 2B gri veya 3 kanalli olmali")
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        height, width = gray.shape[:2]
        scale = min(1.0, self.max_dimension / float(max(height, width)))
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return gray, scale

    def update(
        self,
        gray: np.ndarray,
        timestamp_s: Optional[float],
        gsd_m_per_px: float,
    ) -> OpticalFlowSpeed:
        current, current_scale = self._prepare(gray)
        try:
            timestamp = float(timestamp_s) if timestamp_s is not None else None
        except (TypeError, ValueError):
            timestamp = None
        gsd = float(gsd_m_per_px)

        previous = self._prev_gray
        previous_timestamp = self._prev_timestamp
        previous_gsd = self._prev_gsd_m_per_px
        previous_scale = self._prev_scale

        # Her durumda state'i ilerlet: gecersiz bir olcum sonraki kareyi kilitlemesin.
        self._prev_gray = current
        self._prev_timestamp = timestamp
        self._prev_gsd_m_per_px = gsd if np.isfinite(gsd) and gsd > 0 else None
        self._prev_scale = current_scale

        if previous is None:
            return OpticalFlowSpeed(reason="warmup")
        if previous.shape != current.shape:
            return OpticalFlowSpeed(reason="shape_changed")
        if timestamp is None or previous_timestamp is None:
            return OpticalFlowSpeed(reason="timestamp_missing")
        dt = timestamp - previous_timestamp
        if not np.isfinite(dt) or dt <= 1e-6:
            return OpticalFlowSpeed(dt_s=float(dt) if np.isfinite(dt) else 0.0, reason="invalid_dt")
        if previous_gsd is None or not np.isfinite(gsd) or gsd <= 0:
            return OpticalFlowSpeed(dt_s=dt, reason="invalid_gsd")

        points0 = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if points0 is None or len(points0) < self.min_tracks:
            return OpticalFlowSpeed(dt_s=dt, tracked_points=0, reason="few_features")

        points1, status, _errors = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            points0,
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if points1 is None or status is None:
            return OpticalFlowSpeed(dt_s=dt, reason="flow_failed")

        keep = status.reshape(-1).astype(bool)
        source = points0.reshape(-1, 2)[keep]
        target = points1.reshape(-1, 2)[keep]
        finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        source = source[finite]
        target = target[finite]
        tracked = int(len(source))
        if tracked < self.min_tracks:
            return OpticalFlowSpeed(dt_s=dt, tracked_points=tracked, reason="few_tracks")

        transform, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold_px,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if transform is None or inliers is None:
            return OpticalFlowSpeed(dt_s=dt, tracked_points=tracked, reason="model_failed")

        inlier_ratio = float(np.mean(inliers.reshape(-1).astype(bool)))
        if inlier_ratio < self.min_inlier_ratio:
            return OpticalFlowSpeed(
                dt_s=dt,
                tracked_points=tracked,
                inlier_ratio=inlier_ratio,
                reason="low_inlier_ratio",
            )

        center = np.array([previous.shape[1] / 2.0, previous.shape[0] / 2.0, 1.0])
        moved_center = transform @ center
        displacement_scaled_px = float(np.linalg.norm(moved_center - center[:2]))
        # Iki karenin GSD'sini kendi kucultme oranlarinda ortalayarak metreye cevir.
        previous_scaled_gsd = previous_gsd / previous_scale
        current_scaled_gsd = gsd / current_scale
        displacement_m = displacement_scaled_px * (previous_scaled_gsd + current_scaled_gsd) / 2.0
        speed_mps = displacement_m / dt
        if not np.isfinite(speed_mps):
            return OpticalFlowSpeed(dt_s=dt, tracked_points=tracked, reason="non_finite_speed")

        return OpticalFlowSpeed(
            speed_mps=float(speed_mps),
            displacement_px=displacement_scaled_px,
            dt_s=float(dt),
            tracked_points=tracked,
            inlier_ratio=inlier_ratio,
            valid=True,
            reason="ok",
        )
