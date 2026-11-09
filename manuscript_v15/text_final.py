# -*- coding: utf-8 -*-
"""Results-dependent paragraphs of v15. Placeholders {name} are filled from the numbers dict N
(see fill_extra.py). Keyboard hyphens only."""

FAILURE_PAR = (
    "Three sources account for most of the remaining errors and failures. First, a constant camera offset: in the "
    "body frame of the vehicle the accepted fixes are displaced by about {b_l1d_f} m along and {b_l1d_r} m across the "
    "heading with the Mavic 2 Pro and by about {b_fc_f} m and {b_fc_r} m with the Mavic 2 Zoom, on every route and with "
    "grayscale NCC as well as with the learned representations, which points to a small attitude (boresight) offset of "
    "the camera or a timing offset of the position tag rather than to the matcher. Estimating the offset of each route "
    "from the other doctoral routes flown with the same camera and subtracting it raises the precision of the accepted "
    "fixes of the benchmark-selected model from {p_before}% to {p_after}% on the doctoral routes ({pa_before}% to "
    "{pa_after}% on the additional flights) and lowers their median error from {med_before} m to {med_after} m; the low "
    "success rate of Urban 1 in Table 4 is due to this offset, which exceeds the 30 m threshold at its flight height. "
    "The tables report uncorrected values. Second, heading: the flight-controller yaw used for north alignment differs "
    "from the gimbal yaw by 2-13° depending on the route, and neither source is consistently better (on a subset of "
    "frames, the gimbal yaw would have raised the oracle success of Model 4 on Urban 8 and lowered it on Rural 1); Table 5 "
    "quantifies how quickly heading errors of 5-10° reduce the success rate. Third, terrain and texture: over the hills "
    "of Rural 2 the frames shrink below a 544 px crop at map resolution ({r2_inel} of 211 frames are not eligible), and "
    "homogeneous agricultural parcels cause most of the unresolved rural frames."
)

TEXTURE_PAR = (
    "Low-texture queries. The benchmark gate rejects tiles with a gray-level standard deviation below 12 or an entropy "
    "below 4 bits. To measure performance exactly where the gate applies, a regular 500 m grid over the district "
    "mosaics was classified with the same gate, and all low-texture grid points ({n_low_g} on the Google and {n_low_b} "
    "on the Bing mosaic, excluding missing-data tiles) were matched together with {n_reg} randomly drawn regular points, "
    "with the flight-replay matcher in a ±1 km window around the true position (25 m threshold, clean tiles). On "
    "regular tiles grayscale NCC, gradient magnitude and the CAE reach {reg_gray}%, {reg_sobel}% and {reg_cae}% (mean of "
    "both directions); on low-texture tiles they reach {low_gray}%, {low_sobel}% and {low_cae}%. {texture_sentence}"
)

RECON_PAR = (
    "Cartographic supervision versus input reconstruction. To isolate the training target, the current training "
    "recipe of the authors' pipeline (seven-layer f32-k3 encoder-decoder without spatial reduction, sigmoid output, "
    "binary cross-entropy, batch 1, 12 epochs, inputs scaled to [0, 1], identical training tiles) was trained twice in "
    "the same session with the target as the only difference: the cartographic rendering, or the input satellite tile "
    "itself (an autoencoder that reconstructs its input). The checkpoint of each run was chosen by the same "
    "false-peak gate on a training region and evaluated with 1,000 queries per direction on the district mosaics. "
    "Global success at 25 m (mean of both directions and both variants) is {recon_map}% for the map-supervised network, "
    "{recon_rec}% for the reconstruction network and {recon_raw}% for grayscale NCC. {recon_sentence}"
)

LIT_PAR = (
    "Comparison with the literature. The results are consistent with the season-invariant transform of Fragoso et "
    "al.^{{17}} and the dense season-invariant features of Kinnari et al.^{{19}} in showing that learned pre-processing "
    "can rescue classical matching. The closest method, the OpenStreetMap-supervised translation of Schleiss,^{{20}} was "
    "evaluated on a 560 m × 680 m area at about 300 m above ground; about 20% of the frames could not be matched and the "
    "median error of the remainder was about 40 m (22.7 m on a landmark-rich subset). In the causal replay of the "
    "present work {unres_f48}% of the eligible doctoral frames remain unresolved and the median error of accepted fixes "
    "is {med_f48} m ({med_f48_corr} m after removing the camera offset), over routes of 2-6 km at 300-700 m above "
    "ground; areas, cameras and references differ, so the comparison is indicative only. The present work adds a "
    "transform applied to both the map and the frame, a network two orders of magnitude smaller than a pix2pix "
    "generator, controlled comparisons with grayscale, gradient-magnitude and input-reconstruction representations, a "
    "quantitative account of false-peak suppression and a reusable benchmark with spatial block bootstrap. Compared with "
    "retrieval-style cross-view geolocalization,^{{14-16}} the method returns a metric position at map resolution rather "
    "than a tile identity, at the cost of requiring near-nadir imagery and known heading and scale."
)

VALID_PAR = (
    "Validity boundaries. Several limits must be stated plainly. (i) Geographic scope. Ürgüp and Nevşehir are among "
    "the 125 cities of the training corpus and all evaluations are in the Ürgüp region; the spatial split of the "
    "district benchmark (Section 4.4) separates checkpoint selection from evaluation but not training from evaluation, "
    "so the results are in-domain, and a geographically held-out test with an explicit tile-exclusion buffer is "
    "required before claiming transfer to unseen regions. (ii) Reference accuracy. The consumer GPS reference has an "
    "observed error of 10-15 m and the cameras add a constant offset of 7-22 m (Section 4.2); median errors of about "
    "10-20 m are therefore close to the measurement floor. (iii) Replay, not flight. The causal replay removes GNSS "
    "after the first frame but still processes recorded frames offline; it uses the recorded barometric altitude and "
    "compass heading, initializes from GNSS, propagates the position with a frame-to-frame visual-odometry prior instead "
    "of inertial sensing, and was not run on flight hardware, so latency and power were not measured. (iv) Geometry. "
    "The method assumes near-nadir imagery and a usable heading; heading errors of 5-10° are the dominant sensitivity "
    "(Table 5), and frames whose footprint at map resolution is smaller than a crop (low flights, steep terrain) are not "
    "eligible. (v) Model files. The flown Model 4 file of the doctoral study (epoch 250) was not archived; the earliest "
    "archived checkpoint of the same run (epoch 300) is used and identified by its SHA-256 hash. (vi) Baselines. Learned "
    "matchers (e.g., SuperPoint/SuperGlue or LoFTR) and other learned transforms were not re-implemented; the released "
    "frame manifest, replay program and benchmark allow such comparisons to be added."
)

FUTURE_PAR = (
    "Future work. Four directions follow directly from the results. First, estimate the camera boresight offset in a "
    "short calibration flight and the heading from the imagery itself, the two error sources identified in Section 4.2. "
    "Second, fuse the visual fix with inertial and barometric states in a Kalman or particle filter, using the NCC peak "
    "margin and cross-crop agreement as an adaptive measurement covariance. Third, resolve scale explicitly with a "
    "multi-resolution reference or image-based height estimation^{{51}} so that low and steep-terrain frames remain "
    "eligible. Fourth, extend the benchmark and the flight tests to geographically disjoint regions, seasons and "
    "providers, and enrich the training corpus with rural targets (field boundaries, tracks, watercourses)."
)

INTERP_PAR = (
    "Interpretation. Both evaluation protocols point to the same conclusion: a shallow cross-autoencoder supervised by "
    "cartographic renderings turns plain NCC into a usable terrain-referenced position source. On the same flight "
    "frames it raises the share of frames localized within 30 m from {or_gray}% (grayscale) to {or_f48}% in independent "
    "search, mainly by making the three crops agree more often, and in a causal replay that reads GNSS only at the first "
    "frame it localizes {ca_f48}% of the eligible frames with a precision of {ca_f48_prec}% on the doctoral routes. The "
    "cross-source benchmark, which removes flight-specific confounders and provides thousands of independent queries "
    "with confidence intervals, quantifies the gain as 16.5-19.9 percentage points of global success over a district of "
    "about 500 km², a gain that persists when the checkpoint is selected on one half of the district and evaluated on the "
    "other. The peak-separation probe supplies the mechanism: the representation acts as a learned structural filter "
    "that suppresses perceptual aliasing."
)

STRIDE_PAR = (
    "Stride-1 versus stride-2. The thesis-selected stride-2 architecture compresses the tile to 68 × 68 before decoding. "
    "On the cross-source benchmark the best stride-1 networks of the same family outperform it by a wide margin, and the "
    "evaluated stride-2 checkpoint drops below the grayscale baseline on the district mosaics. On the flight frames the "
    "difference is smaller: in the oracle replay the benchmark-selected stride-1 checkpoint localizes {or_f48}% of the "
    "doctoral frames against {or_m4}% for the stride-2 Model 4 lineage, and in the causal replay {ca_f48}% against "
    "{ca_m4}%. {stride_sentence} For template matching the decoder must reproduce road and building edges at pixel "
    "precision, and a bottleneck that discards spatial detail can hurt a correlation task; the parameter count is not "
    "the driver, and two 50-layer alternatives failed."
)

CONCL_PAR = (
    "We have described a compact cross-autoencoder that converts satellite imagery and nadir UAV imagery into a common "
    "cartographic structural representation, so that normalized cross-correlation with a three-crop consistency check "
    "can localize the vehicle without GNSS. All flight results were regenerated from per-frame records with a released "
    "replay program. On sixteen DJI flights over Ürgüp the learned representation raised the share of frames localized "
    "within 30 m in independent search from {or_gray}% to {or_f48}% on the doctoral routes, and a causal replay that "
    "reads GNSS only at the first frame localized {ca_f48}% of the eligible frames with a precision of {ca_f48_prec}%; "
    "replacing per-frame GNSS altitude by barometric altitude changed the result by {aided_minus_causal_f48} points. A "
    "cross-source map benchmark with thousands of spatially blocked queries, photometric degradation, six search modes "
    "and bootstrap confidence intervals showed that the best shallow stride-1 network raises global success at 25 m from "
    "49% to 69% (clean) and from 46% to 63% (degraded) over a district of about 500 km², that the gain persists under a "
    "spatial split of checkpoint selection and evaluation, and that cartographic supervision {recon_concl}. The "
    "remaining errors are dominated by a constant camera offset and by heading errors, not by the representation. "
    "Within its stated in-domain scope, the method is a practical, explainable and inexpensive candidate visual "
    "position-fix source for GNSS-denied UAV navigation; establishing geographic generalization and closed-loop "
    "performance is the next step."
)

ABSTRACT = (
    "Unmanned aerial vehicles (UAVs) rely on global navigation satellite systems (GNSS), which can be jammed or spoofed. "
    "We present a lightweight cross-autoencoder (50-78 thousand parameters), trained on 406,579 satellite-cartographic "
    "tile pairs from 125 cities, that transforms satellite and UAV images into a shared map-like representation matched "
    "by normalized cross-correlation. Sixteen DJI flights over Ürgüp, Türkiye (2,951 frames) were replayed from "
    "per-frame records with released code. On the same frames the learned representation raised the share localized "
    "within 30 m from {or_gray}% (grayscale) and {or_sobel}% (gradient magnitude) to {or_f48}%; a causal replay that "
    "reads GNSS only at the first frame localized {ca_f48}% of eligible frames with {ca_f48_prec}% precision. A "
    "reproducible cross-source benchmark of Google and Bing mosaics with spatially blocked queries, photometric "
    "degradation and bootstrap intervals raised global success at 25 m over about 500 km² from 49% to 69%, a gain that "
    "held under a spatial split of model selection and evaluation. Remaining errors stem mainly from a constant camera "
    "offset and heading errors. All results are in-domain and offline."
)

DATA_AVAIL = (
    "The code supporting this article is openly available in three repositories: the cross-source benchmark (query "
    "sampling manifests, the hard_v1 degradation profile, matching and block-bootstrap code) at "
    "https://github.com/elestirmen/benchmark; the cross-autoencoder training and inference code at "
    "https://github.com/elestirmen/autoencoder_sat_to_map; and the flight-side localization pipeline at "
    "https://github.com/elestirmen/template-matching, which includes the flight replay program, the scripts that "
    "compute every flight table of this article, the frame manifest of all 2,951 captured frames and the per-frame "
    "records of every replay reported here (eligibility, outcome, estimated position and error of each frame). All "
    "three repositories are dual-licensed under the MIT and Apache-2.0 licenses. Satellite and cartographic imagery "
    "are subject to the license terms of the respective providers and cannot be redistributed; the flight imagery "
    "and the model checkpoints (identified in the records by their SHA-256 hashes) are available from the "
    "corresponding author on reasonable request. The code versions used for the reported results will be archived "
    "as tagged releases with Zenodo DOIs upon acceptance."
)
