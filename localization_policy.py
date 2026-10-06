# -*- coding: utf-8 -*-
"""Benchmark ve bilinen-baslangic GPS-denied veri erisim politikasi."""

from __future__ import annotations

import csv
import math
import os
from typing import Callable, Optional, Tuple

LatLon = Tuple[float, float]
RowCol = Tuple[int, int]
PointXY = Tuple[float, float]


def resolve_camera_altitude_for_dem(
    *,
    sensor_altitude_m: float,
    terrain_center_m: float,
    datum_correction_m: float,
    use_initial_clearance_anchor: bool,
    anchor_sensor_altitude_m: Optional[float] = None,
    anchor_ground_clearance_m: Optional[float] = None,
):
    """DEM geometrisi icin etkin kamera rakimini ve zemin acikligini dondur.

    GPS-denied takipte sonraki yatay konum bilinmez. Bu nedenle tahmin edilen
    merkezdeki mutlak DEM rakimini sensorun mutlak rakimindan cikarmak, takip
    sapinca negatif aciklik uretebilir. Anchor modunda ilk bilinen konumdaki
    aciklik korunur ve sonraki sensor irtifa *degisimi* buna eklenir. Yerel DEM
    yine kose rolyef farklarini belirlemek icin kullanilir.
    """
    sensor = float(sensor_altitude_m)
    terrain = float(terrain_center_m)
    datum = float(datum_correction_m)

    if use_initial_clearance_anchor:
        if anchor_sensor_altitude_m is None or anchor_ground_clearance_m is None:
            anchor_sensor = sensor
            clearance = sensor - terrain + datum
            anchor_clearance = clearance
        else:
            anchor_sensor = float(anchor_sensor_altitude_m)
            anchor_clearance = float(anchor_ground_clearance_m)
            clearance = anchor_clearance + (sensor - anchor_sensor)
        if clearance <= 0.0:
            raise ValueError("kamera-zemin yuksekligi pozitif olmali")
        effective_camera_altitude = terrain - datum + clearance
        return effective_camera_altitude, clearance, anchor_sensor, anchor_clearance

    clearance = sensor - terrain + datum
    if clearance <= 0.0:
        raise ValueError("kamera-zemin yuksekligi pozitif olmali")
    return sensor, clearance, anchor_sensor_altitude_m, anchor_ground_clearance_m


def next_search_roi_size(
    *,
    current_px: float,
    base_px: float,
    maximum_px: float,
    measurement_accepted: bool,
    growth_px: float,
    covariance_target_px: Optional[float] = None,
) -> int:
    """Bir kare sonunda tek arama ROI'sinin boyutunu guncelle.

    Basarisiz bir kare, cagiran kodda kac farkli ret nedeni olursa olsun, en
    fazla ``growth_px`` kadar buyumeye yol acar. Kovaryans bir hedef onerirse
    hedefe de ayni hiz siniriyla yaklasilir; basarisiz karede pencere daralmaz.
    """
    base = max(1, int(round(float(base_px))))
    maximum = max(base, int(round(float(maximum_px))))
    current = max(base, min(int(round(float(current_px))), maximum))
    growth = max(0, int(round(float(growth_px))))

    if measurement_accepted:
        return base

    target = maximum
    if covariance_target_px is not None:
        target = max(base, min(int(round(float(covariance_target_px))), maximum))
        if target <= current:
            return current
    return min(target, current + growth)


def should_reset_search_roi(
    *,
    measurement_valid: bool,
    intersection_mode: str,
    kalman_action: Optional[str],
) -> bool:
    """Guvenilir gorsel kilitte arama ROI'sini taban boyuta dondur.

    Kalman'in uzak ama guvenilir bir ``abc`` eslesmesini ikinci kareyle teyit
    etmek icin ``reacq_wait`` demesi filtre durumunu korur; gorsel arama ROI'si
    ise zaten guvenilir uclu kilit buldugu icin buyuk kalmamalidir.
    """
    if not bool(measurement_valid):
        return False
    if str(intersection_mode).strip().lower() == "abc":
        return True
    return kalman_action not in ("coast_outlier", "coast_unreliable", "reacq_wait")


def select_failure_growth_px(
    *,
    full_growth_px: float,
    tracked_growth_factor: float,
    kalman_tracking_available: bool,
    consecutive_failures: int,
    tracked_failure_frames: int,
) -> int:
    """Takip hala mevcutken yumusak, uzun kayipta normal ROI adimi sec."""
    full = max(1, int(round(float(full_growth_px))))
    if (
        kalman_tracking_available
        and int(consecutive_failures) <= max(0, int(tracked_failure_frames))
    ):
        factor = max(0.0, min(1.0, float(tracked_growth_factor)))
        return max(1, int(round(full * factor)))
    return full


def should_run_global_recovery(
    *,
    benchmark: bool,
    enabled: bool,
    search_anchor_available: bool,
    consecutive_failures: int,
    minimum_failure_frames: int,
    minimum_window_px: int,
    current_window_px: int,
    reference_shape,
) -> bool:
    """Global kurtarmayi ancak ROI kademeli esige ulastiginda ac."""
    if benchmark or not enabled or not search_anchor_available or reference_shape is None:
        return False
    map_span = max(int(reference_shape[0]), int(reference_shape[1]))
    recovery_window = min(map_span, max(1, int(minimum_window_px)))
    return bool(
        int(consecutive_failures) >= max(1, int(minimum_failure_frames))
        and int(current_window_px) >= recovery_window
    )


def load_motion_series(csv_path: str):
    """Bagimsiz hareket sidecar'ini yukle.

    CSV sutunlari: ``image_path,east_m,north_m``. Degerler onceki kareden bu
    kareye yatay deplasmandir; GNSS koordinati degildir. Hem tam yol hem dosya
    adi anahtari uretilir.
    """
    values = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image_path = str(row.get("image_path", "")).strip()
            if not image_path:
                raise ValueError("Motion CSV satirinda image_path eksik")
            delta = (float(row["east_m"]), float(row["north_m"]))
            values[image_path] = delta
            values[os.path.abspath(image_path)] = delta
            values[os.path.basename(image_path).lower()] = delta
    return values


def select_external_motion_pixels(
    *,
    image_path: str,
    external_motion,
    map_resolution_m_per_px: float,
):
    """Dogu/kuzey metre deplasmanini harita ``(dx,dy)`` pikseline cevir."""
    resolution = float(map_resolution_m_per_px)
    if resolution <= 0.0:
        raise ValueError("Harita cozunurlugu pozitif olmali")
    candidates = (image_path, os.path.abspath(image_path), os.path.basename(image_path).lower())
    for key in candidates:
        if key in external_motion:
            east_m, north_m = external_motion[key]
            return (
                float(east_m) / resolution,
                -float(north_m) / resolution,
            ), "external_motion_csv"
    raise KeyError(f"Harici hareket bulunamadi: {image_path}")


def choose_visualization_view(
    *,
    benchmark: bool,
    predicted_xy: PointXY,
    ground_truth_xy: Optional[PointXY],
    mode: str,
    base_half_size_px: int,
    maximum_half_size_px: int,
    edge_margin_px: int,
):
    """Yalniz evaluator gorunumu icin merkez ve yari boyut sec.

    Bu fonksiyonun sonucu lokalizasyon/ROI/Kalman hesaplarina verilmemelidir.
    ``auto_fit`` normalde tahmini merkezde tutar; gercek arac guvenli kadrajdan
    cikacaksa iki noktayi birlikte sigdirir. Ikisini bellek siniri icinde
    sigdirmak olanaksizsa gercek arac gorunur kalsin diye evaluator merkezine
    gecer ve bu durumu kaynak etiketiyle acikca bildirir.
    """
    predicted = (float(predicted_xy[0]), float(predicted_xy[1]))
    ground_truth = (
        (float(ground_truth_xy[0]), float(ground_truth_xy[1]))
        if ground_truth_xy is not None
        else None
    )
    base_half = max(1, int(base_half_size_px))
    maximum_half = max(base_half, int(maximum_half_size_px))
    margin = max(0, min(int(edge_margin_px), base_half - 1))
    selected_mode = str(mode).strip().lower()
    if selected_mode not in {"prediction", "ground_truth", "auto_fit"}:
        raise ValueError(
            "VISUALIZATION_CENTER_MODE prediction, ground_truth veya auto_fit olmali"
        )

    if benchmark and ground_truth is not None:
        return ground_truth, base_half, "oracle_ground_truth"
    if ground_truth is None or selected_mode == "prediction":
        return predicted, base_half, "prediction"
    if selected_mode == "ground_truth":
        return ground_truth, base_half, "evaluator_ground_truth"

    dx = abs(ground_truth[0] - predicted[0])
    dy = abs(ground_truth[1] - predicted[1])
    if dx <= base_half - margin and dy <= base_half - margin:
        return predicted, base_half, "prediction"

    required_half = int(math.ceil(max(dx, dy) / 2.0 + margin))
    if required_half <= maximum_half:
        midpoint = (
            (predicted[0] + ground_truth[0]) / 2.0,
            (predicted[1] + ground_truth[1]) / 2.0,
        )
        return midpoint, max(base_half, required_half), "evaluator_auto_fit"

    return ground_truth, base_half, "evaluator_ground_truth_fallback"


def select_localizer_ground_truth_rc(
    *,
    benchmark: bool,
    strict_gps_denied: bool,
    tracking_seeded: bool,
    ground_truth_rc: Optional[RowCol],
) -> Optional[RowCol]:
    """GT pikselini yalniz oracle/legacy veya strict modun t=0 tohumu icin ac."""
    if benchmark or not strict_gps_denied or not tracking_seeded:
        return ground_truth_rc
    return None


def load_altitude_series(csv_path: str):
    """`image_path,altitude_m` CSV'sini hem tam yol hem basename anahtariyla yukle."""
    values = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image_path = str(row.get("image_path", "")).strip()
            if not image_path:
                raise ValueError("Altitude CSV satirinda image_path eksik")
            altitude_m = float(row["altitude_m"])
            values[image_path] = altitude_m
            values[os.path.basename(image_path).lower()] = altitude_m
    return values


def select_inference_altitude(
    *,
    benchmark: bool,
    strict_gps_denied: bool,
    altitude_source: str,
    observed_exif_altitude_m: Optional[float],
    initial_altitude_m: Optional[float],
    image_path: str,
    external_altitudes=None,
):
    """Localizer irtifasini sec ve kullanilan kaynagi acikca dondur."""
    if benchmark or not strict_gps_denied:
        if observed_exif_altitude_m is None:
            raise ValueError("EXIF altitude gerekli")
        observed = float(observed_exif_altitude_m)
        return observed, "exif_altitude"
    source = str(altitude_source).strip().lower()
    if source == "initial_hold":
        if initial_altitude_m is None:
            raise ValueError("initial_hold icin ilk irtifa gerekli")
        return float(initial_altitude_m), "initial_altitude_hold"
    if source == "exif_altitude_proxy":
        if observed_exif_altitude_m is None:
            raise ValueError("exif_altitude_proxy icin EXIF altitude gerekli")
        observed = float(observed_exif_altitude_m)
        return observed, "exif_altitude_proxy"
    if source == "external_csv":
        external_altitudes = external_altitudes or {}
        candidates = (image_path, os.path.abspath(image_path), os.path.basename(image_path).lower())
        for key in candidates:
            if key in external_altitudes:
                return float(external_altitudes[key]), "external_altitude_csv"
        raise KeyError(f"Harici irtifa bulunamadi: {image_path}")
    raise ValueError(f"Bilinmeyen altitude_source: {altitude_source}")


def validate_inference_policy(
    *,
    benchmark: bool,
    strict_gps_denied: bool,
    use_exif_motion_search_prior: bool,
    use_gps_revert: bool,
) -> str:
    if benchmark:
        return "oracle_gps_roi"
    if strict_gps_denied:
        if use_exif_motion_search_prior:
            raise ValueError(
                "Strict GPS-denied modunda EXIF GPS hareket onseli kullanilamaz"
            )
        if use_gps_revert:
            raise ValueError(
                "Strict GPS-denied modunda GPS geri donusu kullanilamaz"
            )
        return "known_start_gps_denied"
    return "legacy_visual_tracking"


def choose_dem_query_latlon(
    *,
    benchmark: bool,
    strict_gps_denied: bool,
    ground_truth_latlon: Optional[LatLon],
    search_center_rc: RowCol,
    rc_to_lon_lat: Callable[[int, int], Tuple[float, float]],
) -> LatLon:
    """DEM sorgu konumunu sec; strict modda ground truth'u tamamen yoksay."""
    if benchmark or not strict_gps_denied:
        if ground_truth_latlon is None:
            raise ValueError("Benchmark/legacy modunda GPS ground truth gerekli")
        return (float(ground_truth_latlon[0]), float(ground_truth_latlon[1]))
    longitude, latitude = rc_to_lon_lat(
        int(search_center_rc[0]), int(search_center_rc[1])
    )
    return (float(latitude), float(longitude))
