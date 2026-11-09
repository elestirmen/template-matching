# -*- coding: utf-8 -*-
"""Point-by-point response (filled with the numbers dict N from fill_extra.py; keyboard hyphens only)."""

INTRO = (
    "We thank both reviewers for an exceptionally careful reading. Their arithmetic check of the flight tables was "
    "correct: the flight statistics of the previous version were copied from summary tables of the doctoral study and "
    "could not all describe the same frames. In tracing the cause we found that the archived statistics had been "
    "produced with a 70 m success threshold (not 30 m), that the true-negative class was defined by the absence of a "
    "three-crop intersection (not by the search window), that the error statistics had been computed over all frames "
    "rather than over resolved frames, and that the route code of the doctoral study reset the position to the "
    "previous estimate when an error above 300 m was detected against GPS. Because the per-frame records behind those "
    "tables were not archived, we did not attempt to repair them. Every flight result of the revised manuscript was "
    "regenerated from scratch, frame by frame, with a single replay program; every flight table is computed by a "
    "script from its per-frame records, and the program, the scripts, the records and a manifest of all 2,951 "
    "captured frames are released (Data Availability). Section, table and figure numbers below refer to the revised "
    "manuscript."
)

R1 = {
    1: (
        "Agreed; the tables could not be reconciled. They have been removed together with the F1 scores. Table 3 now "
        "reports, for every replay and representation, the number of eligible frames, the success rate, the "
        "availability, the precision and the error statistics, and Table 4 lists the causal replay route by route; both "
        "are computed from one set of per-frame records, and the per-route counts and error statistics of every run are "
        "released. The table script checks the "
        "Cauchy-Schwarz bound used by the reviewer (RMSE of the accepted fixes at least sqrt(S_wrong^2 / (n_wrong n_acc))) "
        "for every route and every run; there is no violation. The archived simulation-stage results of the doctoral "
        "study are mentioned only as the history of model selection (Section 3.7) and are not used as evidence."
    ),
    2: (
        "Section 3.7 (Outcomes and metrics) now defines every quantity. A frame is eligible when, at its true position "
        "and altitude, its map-scaled footprint contains a crop, lies on the reference raster and has DEM coverage; the "
        "same eligible frames are used for every representation and replay mode, so all rows of Tables 3 and 4 share one "
        "denominator, and tracking failures that prevent matching altogether count as unresolved. An eligible frame "
        "yields an accepted fix or remains unresolved; an accepted fix is correct within 30 m. The headline figure is "
        "the success rate (correct / eligible), in which an unresolved frame is a failure; availability (accepted / "
        "eligible) and precision (correct / accepted) are reported separately, and error statistics are computed over "
        "accepted fixes only. The reason for every non-eligible frame is recorded in the manifest."
    ),
    3: (
        "Three changes separate selection from evaluation. (i) The benchmark-selected network (f48-k4-s1, checkpoint "
        "1850) was chosen on satellite mosaics only and had never been applied to flight imagery; all sixteen flights "
        "are therefore an independent test for it, and it is replayed on exactly the same frames as the thesis model "
        "(Table 3). (ii) Within the benchmark the checkpoint selection was repeated with a spatial split: selecting on "
        "the northern half of the district and evaluating on the southern half (separated by a 1 km buffer) gives a "
        "gain of 20.8 points of global success (95% block-bootstrap interval 18.3-23.5) on clean and 16.9 points "
        "(14.2-19.6) on degraded queries, and the reverse split gives 19.3 and 14.7 points (Section 4.4). (iii) For the "
        "thesis model the eleven doctoral routes are stated to be in-sample with respect to model selection, and five "
        "additional flights not reported in the doctoral study are reported separately. Because the archive does not "
        "record whether any of their frames entered Simulation 3, we describe them as additional flights and do not "
        "claim a strict hold-out for that model. The few tracking parameters (gate, window growth, visual-odometry "
        "thresholds) were fixed on the first 80 frames of Urban 5 and on Urban 1 before the full replay and not changed "
        "afterwards, so these two routes are not independent for them. All results remain in-domain (Section 5, "
        "Validity boundaries)."
    ),
    4: (
        "Agreed; the GNSS-aided replay of the doctoral study scaled each frame with its own GNSS altitude and DEM value. "
        "The revised manuscript adds a causal replay (Sections 3.4 and 3.7) that reads GNSS only at the first frame of a "
        "route: the height above ground is the take-off altitude plus the barometric relative altitude minus the DEM "
        "value at the predicted position, the position is propagated from frame to frame by a visual-odometry prior "
        "computed from the images, and an accepted fix must fall inside a gate around the prediction. On the eleven "
        "doctoral routes the three cross-autoencoder checkpoints localize {ca_range}% of the eligible frames within 30 m "
        "in this replay, with a precision of {ca_prec_range}%, against {ca_gray}% for grayscale NCC and {ca_sobel}% for "
        "gradient magnitude (Table 3). Compared with the GNSS-aided replay the success rate falls by "
        "{aided_minus_causal_m4} points for the Model 4 lineage, mostly on Rural 1 and Rural 2, where the predicted "
        "position drifts after a run of unresolved frames and the ground height read under it makes the frames too "
        "small to be matched (Section 4.2). Table 5 reports the "
        "sensitivity of the causal replay to heading errors of 5-10°, height errors of 5-10% and initial position errors "
        "of 200-500 m. We state plainly that this is an offline replay of recorded frames that still uses the recorded "
        "compass heading and barometric altitude, and that latency and power were not measured on flight hardware "
        "(Section 5)."
    ),
    5: (
        "The OpenStreetMap-supervised translation of Schleiss (2019) is now discussed in the related work and compared "
        "quantitatively in Section 5. Two baselines were added that use exactly the same matcher, windows, consistency "
        "rule and frames: untransformed grayscale NCC and a classical structural representation (gradient magnitude of "
        "the smoothed image). A further control trains the same network recipe twice with the target as the only "
        "difference - the cartographic rendering or the input itself - to test whether cartographic supervision "
        "matters (Section 4.5). The gradient-magnitude baseline changed the conclusions of the article. Run through the "
        "district benchmark on the identical 4,000 queries per direction (Section 4.3), it localizes {sb_sob_c}% of "
        "clean and {sb_sob_h}% of degraded queries within 25 m in global search, against {sb_cae_c}% and {sb_cae_h}% "
        "for the benchmark-selected cross-autoencoder and {sb_gray_c}% and {sb_gray_h}% for grayscale; the paired "
        "difference of {sb_d_cae_c} and {sb_d_cae_h} points (95% block-bootstrap intervals {sb_ci_cae_c} and "
        "{sb_ci_cae_h}) holds in both matching directions, on both halves of the spatial split and at every texture "
        "level, and it exceeds every checkpoint of the sweep (Table 7). On the doctoral routes gradient magnitude lies "
        "between grayscale and the best learned checkpoints; on the additional flights and in the local low-texture "
        "probe it is as good as or better than the learned representation. We have therefore reframed the article: "
        "the title, abstract, introduction, Section 4.3, the discussion and the conclusions now present a controlled "
        "comparison between a learned and a classical structural representation and state that the gain of the "
        "learned representation is a gain over raw intensity, not over a classical structural filter. Learned matchers "
        "(SuperPoint/SuperGlue, LoFTR) were not re-implemented; the released frame manifest, replay program and "
        "benchmark make such comparisons straightforward to add, and we list this as a limitation."
    ),
}

R2 = {
    1: (
        "Agreed. Unresolved frames are no longer counted as true negatives. They count as failures in the success rate "
        "(correct / eligible), and the share of frames with an accepted fix is reported separately as availability."
    ),
    2: (
        "Agreed. The true-negative class and the F1 scores have been removed. The GPS-centred search of the doctoral "
        "study is retained only as an explicitly labelled oracle mode (each frame searched independently in a 3 x 3 km "
        "window around its GPS position), which isolates the representation from tracking; it is not presented as "
        "navigation. All statistics were recomputed from per-frame records, which are released with the manifest."
    ),
    3: (
        "See our answer to Reviewer 1, point 4. Replacing the per-frame GNSS altitude and GNSS-position DEM look-up by "
        "barometric altitude and the predicted position lowers the success rate by {aided_minus_causal_m4} points for the "
        "Model 4 lineage and by {aided_minus_causal_f48} points for checkpoint 1850; the tracking loop uses no GNSS after "
        "the first frame. Table 5 quantifies the sensitivity to heading, height and initialization errors."
    ),
    4: (
        "Both networks are now replayed on exactly the same frames with the same code (Tables 3 and 4), together with "
        "grayscale and gradient-magnitude baselines and a second checkpoint of the benchmark lineage. The result is "
        "reported as it came out: in the oracle replay the Model 4 lineage localizes {or_m4}% of the eligible doctoral "
        "frames, f48-k4-s1 checkpoint 1850 (the benchmark-selected model of Table 6) {or_f48}% and checkpoint 1550 "
        "{or_c1550}%, against {or_gray}% for grayscale; in the causal replay the three reach {ca_m4}%, {ca_f48}% and "
        "{ca_c1550}%, against {ca_gray}%. The benchmark ranking of checkpoints therefore does not carry over to the "
        "flight frames, and we now say so explicitly (Section 5, 'Benchmark ranking versus flight ranking') and no "
        "longer use the benchmark to make claims about the flown model. The flown epoch-250 file of the doctoral study was "
        "not archived; the replay uses the earliest archived checkpoint of the same run (epoch 300), identified by its "
        "SHA-256 hash. The flight gains of the learned representation over grayscale are smaller than the benchmark "
        "gains and depend on the checkpoint, and against the gradient-magnitude baseline the learned representation has "
        "no gain on the benchmark at all (Section 4.3); the abstract and conclusions state this."
    ),
    5: (
        "The quality gate is now quantified and its effect measured. On a regular 500 m grid over the district mosaics "
        "the gate rejects 1.8% of the Google and 8.3% of the Bing tile positions for low texture (and 3.5-3.9% as "
        "missing data). All low-texture grid points were then matched, together with regular points drawn from the same "
        "grid, with the flight-replay matcher (Section 4.4): {texture_answer}"
    ),
}
