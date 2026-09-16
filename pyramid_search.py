# -*- coding: utf-8 -*-
"""Coarse-to-fine template matching yardimcilari (cok adayli kaba arama).

Amac
----
Ana betikteki `_match_pyramid()` kaba asamada YALNIZCA tek tepe (top-1) aliyor
ve ince aramayi bu tek tohumun etrafindaki dar ROI ile sinirliyor. Kaba tepe
yanlissa (kendine benzer arazi dokusu, tekrarli vadi/peribacasi yapisi) ince
asama bunu ASLA toparlayamaz; dogru eslesme `-1e9` dolgusu altinda kalir.

Bu modul, benchmark/geospatial_model_benchmark.py icindeki `coarse_to_fine_search`
yaklasiminin cekirdegini alir: kaba asamada NMS ile TOP-K aday secilir, her aday
tam cozunurlukte ayri ayri incelenir, sonuc tek bir tam boy response haritasinda
birlestirilir. Boylece `cv2.minMaxLoc` en iyi konumu adaylar ARASINDAN secer.

Sozlesme
--------
`coarse_to_fine_topk()` ana betigin bekledigi seyi dondurur: `matchTemplate` ile
ayni sekle sahip (H-h+1, W-w+1) float32 response haritasi. Incelenmeyen bolgeler
`fill_value` (varsayilan -1e9) ile doldurulur. Bu sayede cagiran taraftaki
`minMaxLoc` / ofset / kutu cizim mantigi DEGISMEDEN calisir.

`top_k=1` verildiginde davranis mevcut tek adayli yol ile birebir ayni olur.

Bagimliliklar: numpy + opencv (ikisi de ana betikte zaten var).
"""

from __future__ import annotations

import heapq
from collections import namedtuple

import cv2
import numpy as np

__all__ = [
    "Candidate",
    "auto_nms_radius",
    "nms_top_candidates",
    "select_coarse_candidates",
    "refine_candidates",
    "coarse_to_fine_topk",
]

# Kaba asama adayi: response uzayinda (x=sutun, y=satir) ve skor.
Candidate = namedtuple("Candidate", ["x", "y", "score"])

# Incelenmeyen bolgelere yazilan "asla kazanma" degeri (mevcut davranisla ayni).
FILL_VALUE = -1e9


def auto_nms_radius(template_shape) -> int:
    """NMS bastirma yaricapi icin makul varsayilan: template kenarinin yarisi.

    Boylece iki aday en az yarim template kadar ayrik olur; ayni tepenin yan
    yana pikselleri K adayin tamamini tuketemez.
    """
    h, w = template_shape[:2]
    return max(8, int(max(h, w) // 2))


def nms_top_candidates(response, top_k, radius):
    """Response haritasindan greedy kare-NMS ile en iyi `top_k` tepeyi dondurur.

    benchmark/geospatial_model_benchmark.py icindeki blok-heap uygulamasiyla ayni
    mantik: her aday secildikten sonra tum harita yeniden taranmaz, yalnizca
    bastirma penceresine degen 256x256 bloklar yeniden degerlendirilir.
    """
    if response is None or response.size == 0 or top_k <= 0:
        return []
    if top_k == 1:
        # Tek aday icin heap kurmaya gerek yok.
        _, max_value, _, max_location = cv2.minMaxLoc(response)
        if not np.isfinite(max_value):
            return []
        return [Candidate(x=int(max_location[0]), y=int(max_location[1]), score=float(max_value))]

    work = response.astype(np.float32, copy=True)
    block_size = 256
    block_rows = (work.shape[0] + block_size - 1) // block_size
    block_cols = (work.shape[1] + block_size - 1) // block_size
    versions = np.zeros((block_rows, block_cols), dtype=np.int32)
    heap = []

    def push_block(block_y, block_x):
        y0 = block_y * block_size
        x0 = block_x * block_size
        view = work[
            y0 : min(work.shape[0], y0 + block_size),
            x0 : min(work.shape[1], x0 + block_size),
        ]
        if view.size == 0:
            return
        _, max_value, _, max_location = cv2.minMaxLoc(view)
        if not np.isfinite(max_value):
            return
        heapq.heappush(
            heap,
            (
                -float(max_value),
                y0 + int(max_location[1]),
                x0 + int(max_location[0]),
                block_y,
                block_x,
                int(versions[block_y, block_x]),
            ),
        )

    for block_y in range(block_rows):
        for block_x in range(block_cols):
            push_block(block_y, block_x)

    radius = max(1, int(radius))
    candidates = []
    while heap and len(candidates) < top_k:
        negative_value, y, x, block_y, block_x, version = heapq.heappop(heap)
        if version != int(versions[block_y, block_x]):
            continue  # bu blok bastirma sonrasi guncellendi, kayit bayat.
        candidates.append(Candidate(x=int(x), y=int(y), score=float(-negative_value)))
        x0, x1 = max(0, x - radius), min(work.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(work.shape[0], y + radius + 1)
        work[y0:y1, x0:x1] = -np.inf
        first_block_x, last_block_x = x0 // block_size, (x1 - 1) // block_size
        first_block_y, last_block_y = y0 // block_size, (y1 - 1) // block_size
        for affected_y in range(first_block_y, last_block_y + 1):
            for affected_x in range(first_block_x, last_block_x + 1):
                versions[affected_y, affected_x] += 1
                push_block(affected_y, affected_x)
    return candidates


def select_coarse_candidates(
    image,
    template,
    *,
    coarse_scale,
    top_k,
    nms_radius_px,
    method=cv2.TM_CCOEFF_NORMED,
    image_small=None,
):
    """Kucultulmus olcekte arayip adaylari TAM COZUNURLUK response uzayinda dondurur.

    Kaba olcek teknik olarak kurulamiyorsa (template kuculunce haritadan buyuk
    kaliyorsa ya da response bossa) bos liste doner; cagiran taraf merkez
    yedegine duser -- mevcut davranisin aynisi.
    """
    H, W = image.shape[:2]
    h, w = template.shape[:2]
    s = float(coarse_scale)
    if s <= 0:
        return []
    small_W = max(1, int(W * s))
    small_H = max(1, int(H * s))
    small_w = max(1, int(w * s))
    small_h = max(1, int(h * s))
    if small_w > small_W or small_h > small_H:
        return []

    # Cagiran taraf kaba goruntuyu onceden urettiyse (kare basina bir kez)
    # yeniden ornekleme yapma; sonuc ayni, is uc kat az.
    if image_small is None or image_small.shape[:2] != (small_H, small_W):
        image_small = cv2.resize(image, (small_W, small_H), interpolation=cv2.INTER_AREA)
    template_small = cv2.resize(template, (small_w, small_h), interpolation=cv2.INTER_AREA)
    response_small = cv2.matchTemplate(image_small, template_small, method, None)
    if response_small.size == 0:
        return []

    coarse_radius = max(1, int(round(nms_radius_px * s)))
    coarse = nms_top_candidates(response_small, top_k=max(1, int(top_k)), radius=coarse_radius)
    # Kaba koordinati tam cozunurluge geri haritala (mevcut kodla ayni: int(x/s)).
    return [Candidate(x=int(c.x / s), y=int(c.y / s), score=c.score) for c in coarse]


def refine_candidates(
    image,
    template,
    candidates,
    *,
    roi_pad_px,
    method=cv2.TM_CCOEFF_NORMED,
    fill_value=FILL_VALUE,
):
    """Her adayin etrafinda tam cozunurlukte ince arama yapip tek response harmanlar.

    Adaylar response uzayinda (x, y) verilir. Donen harita `matchTemplate` ile
    ayni sekildedir; hicbir adayin penceresine girmeyen konumlar `fill_value`
    kalir.
    """
    H, W = image.shape[:2]
    h, w = template.shape[:2]
    resH, resW = H - h + 1, W - w + 1
    if resH <= 0 or resW <= 0:
        return np.empty((0, 0), dtype=np.float32)

    response_full = np.full((resH, resW), fill_value, dtype=np.float32)
    pad = max(1, int(roi_pad_px))
    wrote_any = False

    for candidate in candidates:
        cx = max(0, min(int(candidate.x), resW - 1))
        cy = max(0, min(int(candidate.y), resH - 1))
        x1 = max(0, cx - pad)
        y1 = max(0, cy - pad)
        x2 = min(resW - 1, cx + pad)
        y2 = min(resH - 1, cy + pad)
        if x2 < x1 or y2 < y1:
            continue
        roi = image[y1 : y2 + h, x1 : x2 + w]
        if roi.shape[0] < h or roi.shape[1] < w:
            continue
        response_roi = cv2.matchTemplate(roi, template, method, None)
        if response_roi.size == 0:
            continue
        target = response_full[y1 : y1 + response_roi.shape[0], x1 : x1 + response_roi.shape[1]]
        # Adaylarin ROI'leri kismen ortusebilir; ortusen bolgede degerler zaten
        # ayni oldugundan maximum ile yazmak hem guvenli hem idempotenttir.
        np.maximum(target, response_roi.astype(np.float32, copy=False), out=target)
        wrote_any = True

    if not wrote_any:
        return np.empty((0, 0), dtype=np.float32)
    return response_full


def coarse_to_fine_topk(
    image,
    template,
    *,
    coarse_scale,
    roi_pad_px,
    top_k=1,
    nms_radius_px=0,
    method=cv2.TM_CCOEFF_NORMED,
    fill_value=FILL_VALUE,
    image_small=None,
):
    """Cok adayli coarse-to-fine template matching; tam boy response dondurur.

    `nms_radius_px <= 0` ise `auto_nms_radius()` kullanilir.
    Kaba asama aday uretemezse haritanin merkezi tek aday olarak alinir
    (mevcut `_match_pyramid` yedegiyle ayni).
    """
    H, W = image.shape[:2]
    h, w = template.shape[:2]
    resH, resW = H - h + 1, W - w + 1
    if resH <= 0 or resW <= 0:
        return np.empty((0, 0), dtype=np.float32)

    if nms_radius_px <= 0:
        nms_radius_px = auto_nms_radius(template.shape)

    candidates = select_coarse_candidates(
        image,
        template,
        image_small=image_small,
        coarse_scale=coarse_scale,
        top_k=max(1, int(top_k)),
        nms_radius_px=nms_radius_px,
        method=method,
    )
    if not candidates:
        candidates = [Candidate(x=resW // 2, y=resH // 2, score=float("nan"))]

    return refine_candidates(
        image,
        template,
        candidates,
        roi_pad_px=roi_pad_px,
        method=method,
        fill_value=fill_value,
    )
