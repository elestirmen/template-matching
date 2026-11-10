# -*- coding: utf-8 -*-
"""Revised benchmark paragraphs (numbers verified from the archived per-query records)."""

QUERY_SAMPLING = (
    "Query sampling. The common extent is divided into 1 × 1 km blocks and query centers are drawn block by block "
    "with a fixed seed (42) and a balanced per-block quota (about five per block in the 300-query runs, 31-32 per "
    "block in the 2,000-query run and up to eight per block in the 4,000-query district runs), so that every block "
    "contributes approximately equally; queries whose 544 × 544 px tile is too dark (more than 20% of pixels at or "
    "below gray level 5, i.e., missing data) or too low in texture (standard deviation < 12 gray levels or entropy "
    "< 4 bits) are rejected and re-drawn, and centers are kept one full tile away from the raster edge. On a regular "
    "500 m grid over the district mosaics this gate rejects {REJ_G} of the Google and {REJ_B} of the Bing tile "
    "positions for low texture (and a further 3.5-3.9% as missing data); the behavior on the rejected low-texture "
    "tiles is measured separately in Section 4.4. The manifest of query centers is fixed and shared by every model, "
    "so all methods see identical queries. Runs with 300, 2,000 and 4,000 queries per direction were performed."
)

BENCH_SOBEL_METHOD = (
    "Gradient-magnitude baseline. The classical structural representation of the flight replay (Section 3.7) is "
    "wrapped as a fixed-weight network - an 11-tap Gaussian of σ = 1.5 px, a 3 × 3 Sobel pair, the magnitude halved "
    "and clipped to 255, within 3.5 gray levels of the OpenCV reference - so that it passes through exactly the "
    "tiling, normalization, pyramid search and bootstrap of the networks. It was added after the first round of "
    "review, at the reviewers' request, and run in global search only, on the same 4,000 queries per direction as "
    "Table 6; the grayscale baseline was recomputed alongside as a check that the queries are identical (Section "
    "4.3). Its smoothing scale was fixed in the flight replay and was not tuned on the benchmark."
)

SELECTION_SPLIT = (
    "Because the final choice of the f48-k4-s1 checkpoint was made on the district data, the selection was repeated "
    "with a spatial hold-out. The 26 block rows of the district were split into a northern half (259 blocks) and a "
    "southern half (239 blocks) separated by one row of 20 blocks (1 km) that was used for neither purpose. Selecting "
    "the best of the 19 checkpoints that were evaluated on the district in both directions (mean global success over "
    "both directions and both query variants) on the northern half gives checkpoint 1550, which on the unseen "
    "southern half reaches {S_CAE_C}% (clean) and {S_CAE_H}% (hard_v1) against {S_RAW_C}% and {S_RAW_H}% for grayscale "
    "NCC, a gain of {S_GAIN_C} points (95% block-bootstrap interval {S_CI_C}) and {S_GAIN_H} points ({S_CI_H}). Selecting "
    "on the southern half gives checkpoint 1850, which on the northern half gains {N_GAIN_C} points ({N_CI_C}) and "
    "{N_GAIN_H} points ({N_CI_H}). The checkpoint chosen on one half is within 0.7 points of the best checkpoint of the "
    "other half in both cases, so the reported advantage does not depend on selecting on the evaluation region. The "
    "architecture family was fixed on the core mosaics before the district runs, and the core does not overlap the "
    "district query blocks of the southern half."
)

SELECTION_SPLIT_VALUES = dict(
    S_CAE_C='56.0', S_CAE_H='49.2', S_RAW_C='35.2', S_RAW_H='32.3', S_GAIN_C='+20.8', S_CI_C='18.3 to 23.5',
    S_GAIN_H='+16.9', S_CI_H='14.2 to 19.6', N_GAIN_C='+19.3', N_CI_C='16.6 to 21.8', N_GAIN_H='+14.7', N_CI_H='11.7 to 17.4')
