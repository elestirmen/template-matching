# -*- coding: utf-8 -*-
"""GPS'siz lokalizasyon kalitesi ve sensor fuzyonu yardimcilari.

Bu modul, `simulasyon` projesindeki ayni adli modulden TASINMISTIR; iki proje
arasinda davranis tutarli kalsin diye fonksiyonlar birebir korunmustur. Saf
Python'dur (yalnizca `math` + `dataclasses`); cv2/numpy/tensorflow gibi agir
bagimlilik ICERMEZ, bu yuzden testlerde dogrudan import edilebilir.

Iceren parcalar:
  - `normalize_match_score`        : Ham TM skorunu (CCOEFF/SQDIFF) [0,1]'e tasir.
  - `LocalizationQuality`          : Kalite sonucu (skor taban/ort, yayilim, guven).
  - `compute_localization_quality` : Uc sablonun skor + geometrik tutarliligindan
                                     kompozit guven ve `is_reliable` bayragi uretir.
  - `fuse_measurement_with_prior`  : Olcumu onceki konumla guvene gore harmanlar;
                                     `max_visual_jump_px`'i asan sicramalari reddeder.

Kutu bicimi (MapBox) = (x, y, w, h); nokta (MapPoint) = (x, y). Ana betikteki
`kare`/`a,b,c` dikdortgenleriyle ayni konvansiyon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

MapBox = Tuple[int, int, int, int]
MapPoint = Tuple[int, int]


@dataclass(frozen=True)
class LocalizationQuality:
    normalized_scores: Tuple[float, ...]
    score_floor: float
    score_mean: float
    center_spread_px: float
    confidence: float
    is_reliable: bool
    reason: str


def distance_between_points(point_a: MapPoint, point_b: MapPoint) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def compute_box_center(box: MapBox) -> MapPoint:
    return (box[0] + (box[2] // 2), box[1] + (box[3] // 2))


def normalize_match_score(score_value: float, is_sqdiff_method: bool) -> float:
    """Ham TM skorunu yontemden bagimsiz olarak [0,1]'e tasir.

    - SQDIFF ailesi: dusuk = iyi -> tersine cevrilir.
    - CCOEFF/CCORR_NORMED: [-1,1] -> [0,1]; normalize degilse pozitif sikistirma.
    """
    score_value = float(score_value)
    if is_sqdiff_method:
        if 0.0 <= score_value <= 1.0:
            return max(0.0, min(1.0, 1.0 - score_value))
        return 1.0 / (1.0 + max(0.0, score_value))

    if -1.0 <= score_value <= 1.0:
        return max(0.0, min(1.0, (score_value + 1.0) / 2.0))

    positive_score = max(0.0, score_value)
    return positive_score / (1.0 + positive_score)


def compute_localization_quality(
    score_values: Sequence[float],
    matched_boxes: Sequence[MapBox],
    predicted_intersection_box: MapBox,
    intersection_mode: str,
    is_sqdiff_method: bool,
    score_threshold: float,
    confidence_threshold: float,
    spread_threshold_px: float,
) -> LocalizationQuality:
    """Uc sablon eslesmesinden kompozit lokalizasyon kalitesi uretir.

    Guven = 0.45*skor_ort + 0.25*skor_taban + 0.20*mekansal_tutarlilik
            + 0.10*kesisim_agirligi. `is_reliable`, uc kapidan (skor tabani,
    merkez yayilimi, guven esigi) gecerse True olur; aksi halde `reason` neden.
    """
    normalized_scores = tuple(
        normalize_match_score(score_value, is_sqdiff_method) for score_value in score_values
    )
    score_floor = min(normalized_scores) if normalized_scores else 0.0
    score_mean = (
        sum(normalized_scores) / float(len(normalized_scores))
        if normalized_scores
        else 0.0
    )

    predicted_center = compute_box_center(predicted_intersection_box)
    if matched_boxes:
        center_spread_px = sum(
            distance_between_points(compute_box_center(box), predicted_center)
            for box in matched_boxes
        ) / float(len(matched_boxes))
    else:
        center_spread_px = float("inf")

    intersection_weight = {
        "abc": 1.00,
        "ab": 0.72,
        "bc": 0.72,
        "ac": 0.72,
        "center_fallback": 0.20,
    }.get(intersection_mode, 0.20)
    spatial_consistency = max(
        0.0,
        1.0 - (center_spread_px / max(1.0, float(spread_threshold_px))),
    )
    confidence = max(
        0.0,
        min(
            1.0,
            (0.45 * score_mean)
            + (0.25 * score_floor)
            + (0.20 * spatial_consistency)
            + (0.10 * intersection_weight),
        ),
    )

    if score_floor < score_threshold:
        reason = "score_floor"
        is_reliable = False
    elif center_spread_px > spread_threshold_px:
        reason = "spread"
        is_reliable = False
    elif confidence < confidence_threshold:
        reason = "confidence"
        is_reliable = False
    else:
        reason = "ok"
        is_reliable = True

    return LocalizationQuality(
        normalized_scores=normalized_scores,
        score_floor=float(score_floor),
        score_mean=float(score_mean),
        center_spread_px=float(center_spread_px),
        confidence=float(confidence),
        is_reliable=is_reliable,
        reason=reason,
    )


def blend_velocity_ema(
    prev_velocity: Tuple[float, float],
    displacement: Tuple[float, float],
    ema_gain: float,
) -> Tuple[float, float]:
    """Hiz tahminini EMA ile gunceller (SAF fonksiyon).

    yeni_hiz = gain*displacement + (1-gain)*prev_velocity (eksen bazli).
    Simulasyon'da predict()'e KOMUT hareketi besleniyordu; offline'da komut yok ->
    ardisik kabul edilen olcumlerin yer degisimi (displacement, px/kare) dronun
    GOZLENEN hizini verir; bu hiz predict()'e beslenince filtre hareketi ONGORUR.

    Parametreler:
      - prev_velocity : onceki hiz (vx, vy) px/kare.
      - displacement  : son kabul edilen olcum yer degisimi (dx, dy) px.
      - ema_gain      : 0..1; buyuk=olcume daha cabuk uyar (gurultulu), kucuk=daha duzgun.
    """
    g = max(0.0, min(1.0, float(ema_gain)))
    return (
        g * float(displacement[0]) + (1.0 - g) * float(prev_velocity[0]),
        g * float(displacement[1]) + (1.0 - g) * float(prev_velocity[1]),
    )


def median_of(values: Sequence[float]) -> float:
    """Medyan (SAF). Bos dizide 0.0 doner. Aykiri degerlere dayanikli orta deger."""
    items = sorted(float(v) for v in values)
    n = len(items)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return items[mid]
    return 0.5 * (items[mid - 1] + items[mid])


def innovation_gate_px(
    position_var_mean: float,
    measurement_var: float,
    sigma: float,
) -> float:
    """Covaryans-tabanli (Mahalanobis) innovation kapisi esigi (px) — SAF fonksiyon.

    Esik = sigma * sqrt(P + R)
      P = konum belirsizligi varyansi (ongoru/predict sonrasi), R = olcum gurultu varyansi.
    Coast'ta P process-noise ile buyur -> kapi GENISLER (kurtarma); iyi takipte update
    P'yi kucultur -> kapi SIKILESIR (outlier reddi). Tek ilkeli buyukluk; sihirli px yok.
    """
    s = max(0.0, float(position_var_mean)) + max(0.0, float(measurement_var))
    return max(0.0, float(sigma)) * math.sqrt(max(0.0, s))


def covariance_window_size_px(
    position_var_mean: float,
    measurement_var: float,
    sigma: float,
    template_px: float,
    base_px: float,
    max_px: float,
) -> float:
    """Covaryans-tabanli arama penceresi boyutu (px), [base_px, max_px] araliginda — SAF.

    Yari-genislik ~ sigma*sqrt(P+R) (gercek konumu yuksek olasilikla icerir); uzerine
    sablon payi (template_px) eklenir. Belirsizlik buyudukce (coast) pencere YUMUSAKCA
    buyur, iyi takipte kuculur -> ad-hoc "aniden MAX" / sabit buyume adimi gerekmez.
    """
    half = max(0.0, float(sigma)) * math.sqrt(
        max(0.0, max(0.0, float(position_var_mean)) + max(0.0, float(measurement_var)))
    )
    size = (2.0 * half) + max(0.0, float(template_px))
    return float(max(float(base_px), min(float(max_px), size)))


def effective_step_gate_px(
    recent_steps: Sequence[float],
    gate_multiplier: float,
    floor_px: float,
    min_samples: int = 3,
) -> float:
    """Adaptif hareket kapisi esigi (px) — SAF fonksiyon.

    Sabit bir px degeri yerine, dronun SON karelerdeki tipik (medyan) adim
    mesafesinin katini sinir olarak kullanir; boylece gercek hareket (medyan adim)
    hep gecer, yalnizca tipikin cok ustundeki sicramalar (yanlis eslesmeler) elenir.
    Kare hizindan/IHA hizindan BAGIMSIZ calisir (otomatik olceklenir).

    Parametreler:
      - recent_steps    : son karelerin ardisik HAM olcum adim mesafeleri (px).
      - gate_multiplier : medyan adimin kac kati sinir olsun ( or. 4.0). <=0 -> adaptif KAPALI.
      - floor_px        : taban (minimum) kapi; yavas/duragan harekette gateti cok
                          daraltmaz. Adaptif KAPALIYKEN mutlak ust sinir olarak kullanilir.
      - min_samples     : adaptif esik icin gereken minimum ornek (warmup).

    Doner: efektif kapi (px). 0 -> "kapi yok" (plan_kalman_measurement_action 0'i KAPALI sayar).
      - gate_multiplier<=0  -> floor_px (0 ise tamamen kapali; >0 ise mutlak ust sinir).
      - yeterli ornek yoksa -> 0 (warmup: gateleme yok, taban hizi ogren).
      - aksi halde          -> max(floor_px, gate_multiplier * medyan(recent_steps)).

    Medyan, pencere icindeki birkac sicramaya (outlier) DAYANIKLIDIR.
    """
    if gate_multiplier is None or float(gate_multiplier) <= 0.0:
        return float(floor_px)

    steps = [float(s) for s in recent_steps]
    if len(steps) < int(max(1, min_samples)):
        return 0.0  # warmup: henuz taban hareketi bilinmiyor -> gateleme yok

    return max(float(floor_px), float(gate_multiplier) * median_of(steps))


def plan_kalman_measurement_action(
    is_reliable: bool,
    is_three_way: bool,
    measurement_distance_px: float,
    max_step_px: float,
    reacquire_jump_px: float,
    reacquire_count: int,
    reacquire_frames: int,
) -> Tuple[str, int]:
    """Bir olcumun Kalman'a nasil uygulanacagini belirleyen SAF karar fonksiyonu.

    Amac: "IHA tek karede isinlanamaz" -> fiziksel olarak imkansiz tek-kare sicramalari
    (yanlis eslesmeler) reddetmek; gercek/kalici konum degisimini ise yalnizca ust uste
    teyit edilen yuksek-guvenli (3'lu) olcumle (re-seed) kabul etmek.

    Parametreler:
      - measurement_distance_px : olcum ile filtrenin ongordugu konum arasi mesafe (px).
      - max_step_px             : makul tek-kare yer degistirme ust siniri (px). 0 = KAPALI.
      - reacquire_jump_px       : klasik re-seed "uzaklik" esigi (px).
      - reacquire_count         : o ana kadarki ardisik "uzak+3'lu" sayaci.
      - reacquire_frames        : re-seed icin gereken ardisik teyit sayisi.

    Doner: (action, new_reacquire_count)
      action:
        "update"            -> makul hareket; olcumle guncelle.
        "reseed"            -> kalici uzak 3'lu teyit edildi; filtreyi olcume re-seed et.
        "reacq_wait"        -> uzak 3'lu ama streak dolmadi; guncelleme yok (sayac artar).
        "coast_outlier"     -> fiziksel kapiyi asan 3'lu-olmayan olcum; reddet (kayip say).
        "coast_unreliable"  -> olcum guvenilmez; coast.

    max_step_px=0 iken: yalnizca klasik (3'lu + reacquire_jump) yol re-seed/wait eder,
    diger tum guvenilir olcumler "update" verir -> MEVCUT DAVRANIS BIREBIR korunur.
    """
    if not is_reliable:
        return "coast_unreliable", 0

    far_step = (float(max_step_px) > 0.0 and float(measurement_distance_px) > float(max_step_px))
    reacq_far = (float(measurement_distance_px) > float(reacquire_jump_px))

    if is_three_way and (reacq_far or far_step):
        reacquire_count = int(reacquire_count) + 1
        if reacquire_count >= int(max(1, reacquire_frames)):
            return "reseed", 0
        return "reacq_wait", reacquire_count
    if far_step:
        return "coast_outlier", 0
    return "update", 0


def fuse_measurement_with_prior(
    prior_center: Optional[MapPoint],
    measured_center: MapPoint,
    quality: LocalizationQuality,
    max_visual_jump_px: float,
    blend_gain: float,
) -> Tuple[MapPoint, bool, float]:
    """Olcumu onceki konumla guvene gore harmanlar.

    - Prior yoksa olcum aynen kabul edilir.
    - Olcum priordan `max_visual_jump_px*1.75`'ten uzaksa veya kalite guvenilmezse
      olcum REDDEDILIR (prior korunur) -> tek-adim yanlis eslesmelerine dayanikli.
    - Aksi halde harmanlama kazanci `blend_gain*confidence` ile olceklenir.

    Doner: (fused_center, kabul_edildi_mi, prior_olcum_mesafesi_px).
    """
    if prior_center is None:
        return measured_center, quality.is_reliable, 0.0

    prior_error_px = distance_between_points(prior_center, measured_center)
    if prior_error_px > (float(max_visual_jump_px) * 1.75):
        return prior_center, False, float(prior_error_px)
    if not quality.is_reliable:
        return prior_center, False, float(prior_error_px)

    effective_gain = max(0.0, min(1.0, float(blend_gain) * quality.confidence))
    fused_center = (
        int(round(prior_center[0] + ((measured_center[0] - prior_center[0]) * effective_gain))),
        int(round(prior_center[1] + ((measured_center[1] - prior_center[1]) * effective_gain))),
    )
    return fused_center, True, float(prior_error_px)
