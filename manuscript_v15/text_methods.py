# -*- coding: utf-8 -*-
"""New or revised paragraphs for v15 (methods, related work). Markup: ^{n} = superscript citation,
*x* = italic. Keyboard hyphens only."""

RELATED_WORK_ADD = (
    "Map-supervised translation. Closest of all is the work of Schleiss,^{20} who trained a conditional "
    "generative adversarial network (pix2pix) to translate aerial images into a three-class map-like "
    "representation (buildings, roads, background) rasterized from OpenStreetMap, aligned the frames with the "
    "heading and altitude of the inertial system and estimated the UAV position by sliding the translated frame "
    "over the map with a sum-of-normalized-squared-differences criterion; on a 560 m × 680 m test area south of "
    "Bonn, about 20% of the frames could not be matched and the median error of the remainder was about 40 m "
    "(22.7 m on a landmark-rich subset). The present work shares the idea of cartographic supervision but differs "
    "in three respects that are tested below: the reference side is transformed by the same network as the query "
    "side, so that matching takes place between two map-like rasters rather than between a translated frame and a "
    "vector map rendering; the network is a seven-layer encoder-decoder with 50-78 thousand parameters, two orders "
    "of magnitude smaller than a pix2pix generator; and the representation is evaluated against untransformed "
    "grayscale and a classical gradient-magnitude representation with the identical matcher, on the same frames and "
    "on the same benchmark queries."
)

SYS_OVERVIEW = (
    "Fig. 1 summarizes the pipeline. Offline, a georeferenced satellite or orthophoto mosaic of the mission area is "
    "tiled, each tile is transformed by the cross-autoencoder, and the transformed tiles are stitched into a "
    "structural reference raster with the same georeferencing (WGS 84 / UTM zone 36N, EPSG:32636; UTM: Universal "
    "Transverse Mercator). Online, each nadir UAV frame is undistorted, rotated to north with the on-board compass "
    "heading, scaled to the map ground sampling distance (GSD) using the height above ground derived from the "
    "barometric altitude and a DEM, transformed by the same network, and matched to the structural reference "
    "raster by NCC. Three overlapping crops of the frame are matched independently and a position is output only "
    "when their matches are geometrically consistent. In operation GNSS is not a matching input; of the evaluation replays only the causal one "
    "reproduces this, whereas the oracle and GNSS-aided replays use the GPS position and GNSS altitude to place and "
    "scale the search (Section 3.7). In the offline replay "
    "reported here (Section 3.7) the GPS position stored in each frame serves as the reference against which "
    "errors are measured and, in the causal replay, is read only once per route, at the first frame, to initialize "
    "the search window and the altitude datum; afterwards the search window follows the previous accepted visual "
    "fix propagated by a visual-odometry prior, the altitude comes from the barometric relative altitude recorded by "
    "the flight controller and the DEM is sampled at the predicted position. A GNSS-aided replay, in which every frame is scaled with its own GNSS "
    "altitude and position, and an oracle replay, in which every frame is searched in a window centered on its own "
    "GPS position, are reported alongside to separate the quality of the representation from the behavior of the "
    "tracking loop."
)

DEM_PAR = (
    "Height above ground is required to scale UAV frames. A 30 m digital elevation model (the Copernicus 30 m "
    "global DEM^{45}) was obtained for both test areas through the DEM Tools module of the Sentinel Application "
    "Platform (SNAP) of the European Space Agency (ESA) and resampled to the grid of the reference mosaic. The "
    "flight height above terrain is the altitude of the vehicle minus the DEM elevation at its position, plus a "
    "constant offset of 26 m for the area, which absorbs the difference between the vertical reference of the "
    "on-board altitude and that of the DEM. In the causal replay the altitude is the barometric relative altitude "
    "recorded by the flight controller added to a datum fixed once at the first frame of the route (the GNSS "
    "altitude minus the relative altitude at that frame), and the DEM is sampled at the predicted position (the "
    "last accepted fix propagated by visual odometry); in the GNSS-aided replay the altitude is the GNSS altitude of the frame and the DEM is sampled at its GNSS "
    "position (Section 3.7). The offset was checked by a scale calibration: for every 20th frame of routes flown with both cameras, the frame "
    "scale that maximizes the NCC score at the GPS position was within 4% of the nominal scale."
)

CAMERA_PAR = (
    "Flight imagery was acquired with two DJI platforms. A Mavic 2 Pro (Hasselblad L1D-20c, 13.2 mm sensor width, "
    "10.26 mm focal length, 5,472 × 3,648 px) flew Urban 4-8, Rural 1-3 and the additional flight Urban 9; a Mavic 2 "
    "Zoom (FC2204, 6.17 mm sensor width, 4.386 mm focal length at the wide end, 4,000 × 3,000 px) flew Urban 1-3 and "
    "the additional flights Urban 10-11 and Rural 4-5 (Section 3.7). Frames are nadir RGB images taken at "
    "approximately 2 s intervals, with GPS position, GNSS altitude, barometric relative altitude and flight yaw in "
    "the image metadata (EXIF and XMP). The Mavic 2 Pro camera was calibrated with Zhang’s method;^{46} the "
    "intrinsic matrix K and distortion vector D (Equations 1-2, OpenCV order k1, k2, p1, p2, k3) are applied to "
    "remove barrel distortion before matching. The Zoom camera was not calibrated and its frames are used without "
    "undistortion; because only the central part of each frame is matched (below), residual distortion is small."
)

ROTATE_PAR = (
    "Each frame is converted to grayscale, rotated to north by the recorded flight yaw with bilinear "
    "interpolation,^{47} and the largest inscribed axis-aligned rectangle is retained. The GSD of the frame is "
    "computed from the sensor width S_{w}, flight height above ground H, focal length f and image width I_{w} "
    "(Equation 3),^{48} and the frame is resampled (area averaging) so that its GSD equals the reference map GSD "
    "of 0.298 m/px. Equation 3 uses the nominal sensor width and focal length from the image metadata; a scale "
    "calibration of Section 3.4 confirmed the nominal scale within 4%."
)

CROPS_PAR = (
    "From the normalized frame three 544 × 544 px crops are extracted - one centered and two displaced "
    "diagonally towards the upper-left and the lower-right by d px along both axes, where d = min(200 px, m) and m "
    "is the margin left by the inscribed rectangle. Each crop is scaled to gray levels in [-1, 1] - the input "
    "normalization of the flight software of the doctoral study and of the benchmark in Section 3.8, without "
    "histogram equalization - transformed by the cross-autoencoder and reduced to its central 512 × 512 px to remove "
    "border artifacts. The reference map is transformed in the same way, tile by tile (544 × 544 px tiles on a 512 px "
    "grid, 16 px removed from every tile edge). Fig. 4(a) shows UAV frames before and after the transformation."
)

NCC_PAR_2 = (
    "Here T̄ is the mean of the template and Īx,y the mean of the reference window under the template at offset "
    "(x, y). The correlation is evaluated coarse-to-fine: the window and the template are reduced by a factor of 4, "
    "the five strongest coarse peaks (non-maximum suppression over half a template) are refined at full resolution "
    "in a ±96 px neighborhood, and the best refined peak is kept. A peak lying on the border of the search window "
    "is discarded, because it usually indicates that the true location is outside the window. Each retained peak "
    "implies a position of the frame center once the known displacement of its crop (0 or ±d px) is removed; crops "
    "whose implied centers agree within 48 px (about 14 m) form the consensus, their mean is the position estimate, "
    "and the number of agreeing crops (two or three) is recorded. If no two crops agree the frame is unresolved and "
    "no position is output. Because the three crops see different parts of the frame, an ambiguous peak of one "
    "crop is unlikely to be confirmed at the geometrically consistent offset by another, so the rule acts as a "
    "cheap consistency check (Fig. 4(b)). The pixel position is converted to UTM coordinates through the affine "
    "transform of the reference raster."
)

TRACKING_PAR = (
    "Search window and tracking. In the route replays the search window is a square of 2,000 px (about 600 m) "
    "centered on a predicted position. The prediction is the last accepted fix propagated from frame to frame by "
    "a visual-odometry prior: the translation between consecutive north-up, map-scaled frames is measured by "
    "phase correlation of their central 1,088 × 1,088 px regions (reduced to 256 × 256 px with a Hann window), "
    "and the last measured translation is reused when the correlation response is below 0.05. This prior replaces "
    "the inertial propagation that a flight system would use; it is computed from the frames alone. The first "
    "frame of a route starts at the initial position (GPS at initialization). A consensus position is accepted "
    "only if it lies within a gate of the prediction; the gate is 300 px (90 m) and grows by 100 px for every "
    "frame since the last accepted fix, up to half the window side (the whole window when no motion prior is "
    "available yet). After a frame without an accepted fix (unresolved or rejected by the gate) the window grows "
    "by 500 px per frame up to 6,000 px, and it returns to 2,000 px after the next accepted fix. No GNSS "
    "information enters this loop. The window size, the consensus tolerance, the gate and the odometry "
    "threshold were fixed on the first 80 frames of Urban 5 and on Urban 1 before the full replay and were not "
    "tuned afterwards. In the oracle replay (Section 3.7) the window is 10,000 px (about 3 × 3 km) centered on "
    "the GPS position of each frame and no gate is applied, so every frame is matched independently."
)

CAMPAIGN_PAR = (
    "Test flights took place in Ürgüp district (Nevşehir, Türkiye) in two areas: an urban area of about 102.5 km² "
    "covering the town and its surroundings and a rural area of about 55.4 km² of agricultural land and valleys "
    "typical of the Cappadocian landscape. Sixteen flights are used (Table 4). Eleven of them, eight urban and "
    "three rural, are the routes of the doctoral study; they were flown between August 2022 and May 2023 under "
    "clear skies at different times of day (including a late-afternoon flight with long shadows) and at different "
    "altitudes, and their frames were also used in the doctoral study to choose Model 4 among 30 candidate "
    "networks (Simulation 3 drew about 1,050 of these frames). The remaining five flights - one urban flight of "
    "May 2023, two short urban flights of August 2022 and two rural flights of December 2020 flown at lower "
    "altitude - were not reported in the doctoral study and were not used in any step of this article before the "
    "replay; the archive does not record whether any of their frames entered Simulation 3, so they are reported "
    "separately as additional flights rather than claimed as a strict hold-out for Model 4. The GPS position stored in each frame is the reference; the consumer-grade receiver of the "
    "platform has an observed horizontal error of roughly 10-15 m, which sets a floor on the measurable "
    "localization error. All 2,951 captured frames of the sixteen flights are listed in a frame manifest with "
    "their metadata and, for every replay, their outcome (released with the code; Data Availability)."
)

PROTOCOL_PAR = (
    "Outcomes and metrics. A frame is eligible when, at its GPS position and GNSS altitude, the normalized frame "
    "contains a 544 px crop (it does not at low height above ground), the frame center lies on the reference raster "
    "and the DEM is defined; the reason for every non-eligible frame is recorded in the manifest. The same eligible "
    "frames are used for every representation and replay mode, so all rows of a table share one denominator. For an "
    "eligible frame the replay either outputs an accepted fix or leaves the frame unresolved: no consensus, rejected "
    "by the gate, or - in tracking - not matched at all because the predicted footprint was smaller than a crop or "
    "left the raster. "
    "An accepted fix is correct if it lies within 30 m of the GPS reference and wrong otherwise. The threshold is "
    "two to three times the observed error of the GPS reference and reflects the uncertainty of the reference and "
    "of the georeferencing chain. For each route we report the success rate (correct fixes divided by eligible "
    "frames, the headline figure, in which an unresolved frame counts as a failure to provide a position), the "
    "availability (accepted fixes divided by eligible frames), the precision among accepted fixes (correct divided "
    "by accepted), and the median, mean absolute error (MAE), root-mean-square error (RMSE) and 90th percentile of "
    "the error over accepted fixes. Errors are great-circle (haversine) distances between true and estimated "
    "coordinates. Thresholded F1 scores are no longer used: the true-negative class of the doctoral study depended "
    "on whether the three-crop intersection existed, not on the search window, and a metric that discards "
    "unresolved frames overstates reliability for continuous localization."
)

MODES_PAR = (
    "Replay modes and models. Every route is replayed in three modes with identical code: *oracle*, in which "
    "each frame is searched independently in a 10,000 px window centered on its own GPS position (the protocol of "
    "Simulation 3 of the doctoral study), which measures the reliability of the representation without any "
    "tracking; *GNSS-aided tracking*, in which the window follows the previous visual fix but the frame is scaled "
    "with its own GNSS altitude and GNSS-position DEM look-up, as in the doctoral study; and *causal tracking*, in "
    "which GNSS is read only at the first frame (Sections 3.1 and 3.4). Three representations are compared with "
    "the same matcher: the cross-autoencoder, untransformed grayscale, and a classical structural representation "
    "(gradient magnitude of the Gaussian-smoothed grayscale image, σ = 1.5 px, applied to map and frame alike). "
    "Two cross-autoencoder lineages are evaluated, with checkpoints frozen before the replay: the thesis lineage f48-k5-s2 "
    "(Model 4 of the doctoral study; the archived checkpoint of epoch 300, the earliest archived checkpoint of the "
    "lineage, is used because the 250-epoch file of the doctoral study itself was not archived) and the "
    "benchmark-selected lineage f48-k4-s1 (checkpoint {F48_CKPT}), which had never been applied to flight imagery "
    "before: its architecture and checkpoint were chosen on satellite mosaics only (Section 4.4), and the "
    "checkpoint was frozen before the replay. Checkpoint 1550 of the same run, which was best on the core mosaics, is "
    "replayed in the oracle and causal modes as a check of checkpoint sensitivity. All sixteen flights are therefore an independent test for the "
    "benchmark-selected model, whereas for Model 4 the eleven doctoral routes are in-sample with respect to model "
    "selection."
)

SIM_PAR = (
    "Before the flight campaign, the doctoral study used three simulation stages to shortlist and select the "
    "network: Simulation 1 localized 512 × 512 px patches of a Bing Maps mosaic of Ürgüp on a Google Maps mosaic of "
    "the same 4 × 4 km area at 256 px steps after both were transformed by each of the 30 best networks (Fig. 5), "
    "the precursor of the benchmark in Section 3.8; Simulation 2 moved a simulated vehicle in 100 px steps along a "
    "20-waypoint route on one mosaic while localizing itself on the other; and Simulation 3 localized about 1,050 "
    "frames of the flight campaign within a 10,000 × 10,000 px section of the reference map re-centered on the GPS "
    "position of each frame, for all 30 networks and for a grayscale baseline. The route statistics of the "
    "doctoral study were archived only as summary tables; the underlying per-frame records are not available and "
    "the archived summaries are not internally consistent (the standard deviations of the rural routes exceed the "
    "corresponding RMSE values, and the confusion counts and error statistics of several routes cannot describe the "
    "same set of frames). They are therefore not reproduced here. All flight results in this article were "
    "regenerated from scratch with the released replay code from the per-frame records it writes."
)

CONTRIB_FLIGHT = (
    "A flight evaluation over Ürgüp, Türkiye, comprising sixteen DJI flights (2,951 frames; eleven routes of the "
    "doctoral study and five additional flights), regenerated from a common per-frame manifest with released code "
    "and reported route by route as success rate over all eligible frames, availability, precision among accepted "
    "fixes and error statistics, in three replay modes including a causal GNSS-denied replay that reads GNSS only "
    "at initialization, with sensitivity analyses for heading, altitude and initialization errors."
)

CONTRIB_BASELINES = (  # placeholder; build_results.py replaces it with the result-dependent wording of narrative.py
    "Controlled comparisons with the identical matcher, on the same flight frames and on the same benchmark queries, "
    "between the learned representation, untransformed grayscale and a classical gradient-magnitude representation, "
    "and between the thesis-selected stride-2 checkpoint and the benchmark-selected stride-1 checkpoint."
)

INTRO_QUESTION = (
    "This article is derived from, and extends, the first author’s doctoral thesis.^{26} The thesis proposed a "
    "cross-autoencoder (CAE): an encoder-decoder network trained to transform a satellite image tile into the "
    "corresponding cartographic (map) rendering of the same coordinates. Because the target is a map rather than the "
    "input itself, the decoder is forced to keep persistent structure - roads, building outlines, parcel boundaries - "
    "and to discard radiometry, shadows and vegetation state, and applying the same network to both the reference "
    "mosaic and the UAV frame yields two map-like images that can be matched with plain NCC. The question of the "
    "present article is whether this learned structural representation improves correlation-based localization "
    "beyond what a classical structural representation - the gradient magnitude of the smoothed image, which needs "
    "no training - already provides, when both are evaluated with the identical matcher on the same data. The "
    "contributions are as follows."
)

TITLE = ("Learned versus Classical Structural Representations for GNSS-Denied UAV Localization by Template Matching: "
         "A Cross-Source Map Benchmark and Flight Replay")

KEYWORDS = ("Keywords: GNSS-denied navigation; UAV localization; template matching; structural representation; "
            "cross-autoencoder; gradient magnitude; cross-source benchmark.")
