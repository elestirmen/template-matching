# UAV Image Positioning - Deep Learning Supported Geolocalization via Template Matching

*[Türkçe sürüm için tıklayın (README.tr.md)](README.tr.md)*

![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-FF6F00)
![OpenCV](https://img.shields.io/badge/opencv-4.10-5C3EE8)
![Status](https://img.shields.io/badge/status-research%20prototype-yellow)
![License](https://img.shields.io/badge/license-all%20rights%20reserved-lightgrey)

This project is an experimental computer vision system that investigates the automated positioning of UAV (Unmanned Aerial Vehicle) images on a **reference orthophoto map**. It combines deep learning-based image transformation with multi-scale template matching, supporting controlled GPS-centric benchmark and sequential visual tracking scenarios as separate operating modes.

> **Research prototype:** The software is intended for academic experiments and method development. It is not a validated product for safety-critical navigation or standalone real flight control. Default parameters are specific to approximately 30 cm/pixel data around Ürgüp; recalibration is required for different regions, altitudes, and sensors.

---

## Table of Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Academic Framework](#academic-framework)
- [System Architecture](#system-architecture)
- [Workflow (Pipeline)](#workflow-pipeline)
- [Repository Layout](#repository-layout)
- [Code Map](#code-map)
- [Configuration (RUN\_CFG)](#configuration-run_cfg)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Alternative Front-Ends](#alternative-front-ends)
  - [Reproducible Experiment Runs](#reproducible-experiment-runs)
  - [Runtime Controls](#runtime-controls)
  - [Tests](#tests)
- [Technical Details](#technical-details)
  - [Reading EXIF Data](#reading-exif-data)
  - [DEM-Based Multi-Scale Approach](#dem-based-multi-scale-approach)
  - [Image Preprocessing](#image-preprocessing)
  - [Keras Model Inference](#keras-model-inference)
  - [Template Matching](#template-matching)
  - [Position Determination (Intersection Method)](#position-determination-intersection-method)
  - [Pyramid Search](#pyramid-search)
  - [CUDA GPU Acceleration](#cuda-gpu-acceleration)
  - [Optical Flow Speed Estimation](#optical-flow-speed-estimation)
  - [Kalman Filter (Position Tracking)](#kalman-filter-position-tracking)
- [Coordinate Transformations](#coordinate-transformations)
- [Evaluation Metrics](#evaluation-metrics)
- [Experiment Protocol and Reproducibility](#experiment-protocol-and-reproducibility)
- [Research Tooling](#research-tooling)
- [Visualization](#visualization)
- [Output Files](#output-files)
- [Camera Support](#camera-support)
- [Limitations and Notes](#limitations-and-notes)
- [Citation](#citation)
- [License](#license)

---

## Overview

The system localizes instantaneous images taken by a drone by searching for them on a georeferenced orthophoto map. The basic idea is:

1. **Yaw (heading angle)**, **GPS coordinates**, **focal length**, and **altitude** information are read from the EXIF metadata of the drone image.
2. Using a Digital Elevation Model (DEM), the **terrain elevation** is determined and the flight altitude is calculated.
3. Using the flight altitude and camera parameters, the **Ground Sample Distance (GSD)** is calculated.
4. The image is aligned to the North according to the yaw angle and scaled according to the GSD.
5. **Template patches** are generated at three different scales (based on center, top-left, and bottom-right elevation values).
6. Each patch is processed with a **Keras deep learning model** to extract a feature map.
7. **Template matching** (Normalized Cross-Correlation) is performed in the relevant region of the reference map.
8. The **final position** is determined from the intersection area of the three matches.
9. The error between the estimated position and the actual GPS position is calculated.

Independently of this position channel, an **optical-flow speed estimator** tracks consecutive frames to report drone ground speed even when no reliable match is available. Optional layers (Kalman filtering, composite quality scoring, sensor fusion, diagnostics) sit on top of the core flow above, each gated by its own `RUN_CFG` flag — see [Configuration](#configuration-run_cfg).

---

## Quickstart

The repository ships with one sample route (`guzergahlar/1_tezde_ucus_5`), one map (`haritalar/`), and one model (`model/`), so the default `RUN_CFG` runs out of the box once dependencies are installed:

```bash
conda create -n visual_navigation -c conda-forge python=3.11 \
    gdal=3.8.4 rasterio=1.3.10 numpy=1.26.4 pandas=2.2.2 \
    pyproj=3.6.1 affine=2.4.0 pillow=10.4.0 piexif
conda activate visual_navigation
pip install opencv-python==4.10.0.84 tensorflow==2.16.1

python template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py
```

This opens the `konum` HUD window, processes every image under `RUN_CFG["ANLIK_DIR"]` against every map/model pair, and writes results to `sonuclar.csv` / `sonuclar.txt` / `modele_gore_sonuclar.txt`. See [Installation](#installation) for Windows GDAL notes, [Configuration](#configuration-run_cfg) to point it at your own data, and [Alternative Front-Ends](#alternative-front-ends) for the Qt and headless entry points.

---

## Academic Framework

### Research Problem

The fundamental problem is to estimate the position of an aerial image on a georeferenced reference map from visual content and to maintain this estimation stably across sequential frames. The project specifically focuses on these research questions:

1. To what extent does GSD, calculated with DEM and camera geometry, reduce the scale difference between the query and the map?
2. Can cross-source matching be performed when a learned image transformation is used together with classical normalized cross-correlation?
3. Does the geometric intersection of three elevation/scale hypotheses produce a more stable position than a single match?
4. Do quality gating, Kalman filtering, and re-acquisition reduce false jumps in sequential frames?
5. Under what conditions do separate velocity channels derived from visual position and optical flow provide consistent results?

### Methodological Contributions

- Three scale hypotheses generated from center, top-left, and bottom-right DEM samples.
- Transformation of `544 × 544 × 1` queries within a single batch using a Keras model.
- Pyramid NCC search on CPU or appropriate OpenCV-CUDA setup.
- Optional quality measure combining match score and geometric spread of candidates.
- Confidence-weighted Kalman update, motion gating, and re-acquisition after loss.
- Optical flow velocity estimation based on Lucas–Kanade + RANSAC, independent of localization speed.
- A pure-Python, unit-tested policy layer (`localization_policy.py`) that *specifies* a stricter GPS-denied evaluation contract, plus reproducible-run manifests (`experiment_tracking.py`). See [Experiment Protocol and Reproducibility](#experiment-protocol-and-reproducibility) for its current integration status.

### Meaning of Experiment Modes

| Mode | Search Center | Appropriate Use | Scientific Boundary |
|---|---|---|---|
| `BENCHMARK=True` | Fixed window centered on EXIF/GPS at each frame | Controlled comparison of model and matcher components | Not an end-to-end GPS-denied autonomy result; bounds the true position search |
| `BENCHMARK=False` | Adaptive window tracking the previous visual/filtered position | Sequential tracking and re-acquisition experiment | Separation of initialization, search, and evaluation information must be verified separately |

In a fair GPS-denied experiment, the true position of future frames must not be used for search center, motion prior, recovery, or filter updates. Actual GPS should only be kept as an evaluation label, and `USE_GPS_REVERT=False` must be set.

---

## System Architecture

```text
+---------------------------------------------------------------------+
|                         DRONE IMAGES                                |
|                        (parcalar/ folder)                           |
+--------------------------------+------------------------------------+
                                 |
                    +------------v------------+
                    |    EXIF Reader          |
                    |  (yaw, GPS, alt,        |
                    |   focal length)         |
                    +------------+------------+
                                 |
              +------------------+------------------+
              |                  |                  |
    +---------v--------+ +------v------+ +---------v---------+
    |  DEM Query       | | GSD Calc    | | Yaw Correction    |
    |  (Find Elev.)    | | (cm/pixel)  | | (Align North)     |
    +---------+--------+ +------+------+ +---------+---------+
              |                  |                  |
              +------------------+------------------+
                                 |
                    +------------v------------+
                    |  Multi-Scale Patch      |
                    |   Generation (3 patches)|
                    |  Top-left / Center /    |
                    |     Bottom-right Elev   |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |   Keras Model           |
                    |  (Feature Extraction)   |
                    |   Batch predict         |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |  Template Matching      |
                    |  (TM_CCOEFF_NORMED)     |
                    |  CPU or CUDA GPU        |
                    |  Pyramid search opt.    |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |  Intersection Analysis  |
                    |  (Combine 3 results)    |
                    +------------+------------+
                                 |
              +------------------+------------------+
              |                  |                  |
    +---------v--------+ +------v------+ +---------v---------+
    | Position Est.    | | Error Calc  | |  Visualization    |
    | (Lat/Lon)        | | (Haversine) | |  (OpenCV HUD)     |
    +------------------+ +-------------+ +-------------------+
```

> The box above still says `parcalar/` for historical reasons; the folder the pipeline actually reads is whatever `RUN_CFG["ANLIK_DIR"]` points to (default: `guzergahlar/1_tezde_ucus_5/`). `parcalar/` is now an empty, legacy placeholder — see [Repository Layout](#repository-layout).

---

## Workflow (Pipeline)

When the script is executed, the following steps are performed sequentially:

### 1. Initialization and Resource Loading
- All parameters are read from the `RUN_CFG` configuration dictionary (plus any `ROUTE_PROFILE_OVERRIDES` matching the current route).
- Map files (`haritalar/`), model files (`model/`), and instantaneous images (`ANLIK_DIR`) are listed.
- The DEM (Digital Elevation Model) raster file is opened using GDAL.
- The CRS (Coordinate Reference System) information of the reference map is read and a `pixel -> coordinate` transformer is prepared.

### 2. Model and Map Loop
For each map-model pair:
- The Keras model (`load_model`) is loaded.
- The reference map is read in grayscale using OpenCV.
- CRS/transform information of the map is obtained with Rasterio.

### 3. Instantaneous Image Processing Loop
For each drone image:

#### 3.1 EXIF Reading
- EXIF data is parsed using the PIL/Pillow library.
- Yaw angle (MakerNote -> FlightDegree), GPS coordinates, altitude, and focal length are extracted.

#### 3.2 DEM Query and GSD Calculation
- The **terrain elevation** is read from the DEM pixel corresponding to the drone's GPS location.
- Elevation values at the top-left and bottom-right corners are also obtained separately.
- `Flight Altitude = GPS Altitude - Terrain Elevation + RAKIM_DUZELTME`
- `GSD (cm/px) = (Sensor Width x Flight Altitude x 100) / (Focal Length x Image Width)`

#### 3.3 Image Preprocessing
- The image is aligned North by rotating it by the **inverse of the yaw angle** (plus `YAW_OFFSET_DEG` calibration).
- The **largest internal rectangle** is cropped from the rotated image (black corners are removed).
- It is scaled according to the GSD ratio to match the reference map resolution.
- `PATCH_SIZE` (544) pixel patches are cropped at three different scales (based on center, top-left, bottom-right elevation).

#### 3.4 Deep Learning Model Inference
- 3 patches are fed to the model as a single batch.
- Histogram equalization and [-1, 1] normalization are applied.
- The model output is converted to the 0-255 range and border pixels (`PRED_BORDER`) are cropped.

#### 3.5 Template Matching
- Matching is performed with `cv2.TM_CCOEFF_NORMED` in the search frame section of the reference map.
- If a CUDA GPU is available, accelerated matching with `cv2.cuda.createTemplateMatching` is used.
- Optional pyramid search: first a coarse search at low resolution, then a fine search in the found region.
- Concurrent matching for 3 templates (`ThreadPoolExecutor`).

#### 3.6 Position Determination
- The **intersection area** between the rectangles of the 3 matching results is calculated.
- Pixel coordinates of the intersection center -> geographic coordinate transformation is performed.
- The distance between prediction and actual is calculated using the Haversine formula.

#### 3.7 Adaptive Search Frame
- In normal mode: the search frame for the next frame is narrowed to the region close to the current prediction (optionally centered on the Kalman-filtered position, see `KALMAN_WINDOW_FOLLOWS`).
- If matching fails, the frame is expanded gradually (`KALMAN_LOST_GROWTH_PX`, bounded by `CERCEVE_BOYUTU_MAX`).
- In benchmark mode: a fixed frame centered around the EXIF GPS is used for every frame.

#### 3.8 Optical Flow Speed (independent channel)
- Regardless of the outcome above, sparse Lucas–Kanade tracking between this frame and the previous one estimates ground speed; see [Optical Flow Speed Estimation](#optical-flow-speed-estimation).

### 4. Result Reporting
- RMSE, MAE, standard deviation are calculated.
- Precision, Recall, and F-score are calculated.
- Results are written to the `sonuclar.csv`, `sonuclar.txt`, and `modele_gore_sonuclar.txt` files.

---

## Repository Layout

Most binary assets (`*.tif`, `*.h5`, drone imagery, sweep/run outputs) and a few newer helper scripts are intentionally git-ignored (see `.gitignore`) and only exist in a local working copy; the tree below is what you actually work with day to day, not the much smaller set of files tracked in git.

```text
template-matching/
├── template_matching_parallel_processing_560_hizli_solust_sagalt_
│   koordinat_fonksiyonlar_icinde_cursor.py   # Main pipeline (run this)
├── gps_denied_autonomy.py                    # Quality / fusion / diagnostics (imported)
├── optical_flow_speed.py                     # Optical-flow speed channel (imported)
├── konum_ui_qt.py                            # PyQt5 control-panel front-end
├── run_localization.py                       # Reproducible-run CLI (standalone — see Code Map)
├── localization_policy.py                    # GPS-denied policy rules (standalone — see Code Map)
├── tracking_filter.py                        # Kalman filter, testable copy (standalone)
├── experiment_tracking.py                    # Run-manifest / frame-accounting utilities (standalone)
├── requirements.txt
│
├── headless/                                 # Ekransiz (headless) Linux runner
│   ├── README.md                             # Setup, systemd service, troubleshooting
│   ├── run_headless.py
│   └── requirements-linux.txt
├── tools/                                    # Kalman parameter-sweep + analysis scripts
├── tests/                                    # unittest suite (8 files — see Tests)
│
├── haritalar/                                # Reference orthophoto map(s) (.tif) — HARITA_DIR
├── model/                                    # Keras model(s) (.h5) — MODEL_DIR
├── guzergahlar/                              # Drone image routes
│   └── 1_tezde_ucus_5/                       # Sample route shipped with the repo — default ANLIK_DIR
├── ana_harita_urgup_30_cm_utm_elevation.tif  # Default DEM — DEM_PATH
│                                              # (additional large DEM/orthophoto GeoTIFFs live at the
│                                              #  project root for the author's own reruns; only DEM_PATH
│                                              #  and the folders above matter for a fresh run)
│
├── results/                                  # Archived manual/sweep run outputs (results/kalman_sweep/, ...)
├── run_artifacts/                            # Reproducible-run manifests + frame CSVs (see below)
├── diagnostics/                              # Triptych PNGs, only when DIAGNOSTIC_ENABLED=True
├── sonuclar.csv / sonuclar.txt               # Latest run: detailed results
├── modele_gore_sonuclar.txt                  # Latest run: summary metrics per model
│
├── arşiv/                                    # Archived snapshot of the main script
├── old/, top_modeller/, anlik/, anlik_t/,    # Legacy / alternate data folders (mostly empty
│   parcalar/, temp/                          #  placeholders today, kept for compatibility)
│
├── README.md / README.tr.md
└── .gitignore
```

---

## Code Map

| Module | Role | Status |
|---|---|---|
| `template_matching_parallel_processing_..._cursor.py` | Main pipeline: EXIF → DEM/GSD → preprocessing → Keras inference → template matching → intersection → Kalman/quality/fusion → visualization → CSV/TXT results. | **Active** — this is what you run. |
| `gps_denied_autonomy.py` | Composite localization quality scoring, sensor fusion blending, diagnostic triptych/CSV writers. | Imported by the main script; gated by `USE_QUALITY` / `USE_FUSION` / `DIAGNOSTIC_ENABLED` (all default `False`). |
| `optical_flow_speed.py` | Lucas–Kanade + RANSAC optical-flow speed estimator, independent of the localization channel. | Imported by the main script; gated by `OPTICAL_FLOW_SPEED_ENABLED` (default `True`). |
| `konum_ui_qt.py` | PyQt5 control-panel front-end (background `QThread` worker, live metric cards, toggles); dynamically loads the main script as a module and reuses its core functions. | Alternative entry point (`python konum_ui_qt.py`); needs `PyQt5` (not in `requirements.txt`). |
| `headless/run_headless.py` | Runs the same pipeline with all OpenCV GUI windows suppressed, for headless Linux servers; periodically dumps annotated PNGs instead. | Alternative entry point — see [headless/README.md](headless/README.md). |
| `run_localization.py` | CLI running two named, reproducible modes (`benchmark`, `simulation`) via `RUN_CFG` overrides injected through `TM_RUN_CFG_JSON`. | Standalone, unit-tested (`tests/test_run_localization.py`). The main script only reacts to the subset of injected keys it already understands. |
| `localization_policy.py` | Pure-Python decision rules for a stricter GPS-denied protocol: altitude-source selection, whether GPS ground truth may reach the DEM query or ROI evaluation, adaptive ROI growth, `validate_inference_policy` guardrails. | Standalone, unit-tested (`tests/test_localization_policy.py`). **Not imported by the main pipeline yet.** |
| `tracking_filter.py` | `ConstantVelocityKalmanFilter` — a clean, testable reimplementation of the Kalman logic embedded in the main script. | Standalone, unit-tested (`tests/test_tracking_filter.py`); the main script keeps its own inline Kalman implementation. |
| `experiment_tracking.py` | Reproducible-run manifest builder: asset fingerprints, environment/package snapshot, git revision + dirty flag, per-frame status accounting with coverage/dropout/recovery stats. | Standalone, unit-tested (`tests/test_experiment_tracking.py`); not imported by the main pipeline yet. |
| `tools/kalman_sweep.py`, `tools/analyze_kalman_sweep.py` | Parameter-sweep runner (spawns the main script per route/config) and result summarizer/ranker for Kalman tuning. | Research tooling, run manually — see [Research Tooling](#research-tooling). |

The "standalone" modules above were ported from a related `simulasyon` (simulation) project and are unit-tested in isolation; wiring them into the main pipeline is tracked as follow-up work, not something already reflected in benchmark numbers. See [Experiment Protocol and Reproducibility](#experiment-protocol-and-reproducibility) for what this means in practice.

---

## Configuration (RUN_CFG)

All runtime parameters are managed from a single `RUN_CFG` dictionary near the top of the main script. After this dictionary is read, it is converted into type-safe constants (bool/int/float). A second, smaller dictionary, `ROUTE_PROFILE_OVERRIDES`, immediately follows it — see [Per-Route Profile Overrides](#per-route-profile-overrides).

### Core Paths, Search & Scale

| Parameter | Default | Description |
|-----------|-----------|----------|
| `BENCHMARK` | `False` | If `True`, uses a fixed frame around EXIF GPS center for every frame (adaptive tracking is disabled) |
| `DEBUG` | `False` | If `True`, intermediate images (patch, model output) are displayed on the screen |
| `PATCH_SIZE` | `544` | Model input size (pixels) |
| `PRED_BORDER` | `16` | Border to crop from the model output (pixels) |
| `USE_PYRAMID` | `True` | Enable pyramid (coarse-to-fine) search |
| `COARSE_SCALE` | `0.5` | Coarse scale factor for pyramid search |
| `ROI_PAD_FACTOR` | `0.4` | Expansion coefficient for the fine search region in pyramid search |
| `CERCEVE_BOYUTU_NORMAL` | `2048` | Search frame size in normal mode (pixels) |
| `CERCEVE_BOYUTU_BENCHMARK` | `5000` | Search frame size in benchmark mode (pixels) |
| `CERCEVE_BOYUTU_MAX` | `15000` | Maximum search-frame size allowed during adaptive growth (pixels) |
| `FARK_MAX` | `200` | Maximum pixel clamp applied when shifting a patch position |
| `USE_EXIF_MOTION_SEARCH_PRIOR` | `False` | If `True`, shifts the next search-window center using pixel motion implied by consecutive EXIF GPS fixes, without exposing GPS as the reported output position (a softer prior than `USE_GPS_REVERT`) |
| `HARITA_DIR` | `"haritalar"` | Folder containing map files |
| `MODEL_DIR` | `"model"` | Folder containing Keras model files |
| `ANLIK_DIR` | `"guzergahlar/1_tezde_ucus_5"` | Folder containing drone images |
| `DEM_PATH` | `"ana_harita_urgup_30_cm_utm_elevation.tif"` | DEM raster file path |
| `HARITA_DOSYALARI` | `[]` | List of specific map files (empty = all in folder) |
| `MODEL_DOSYALARI` | `[]` | List of specific model files (empty = all in folder) |
| `SORT_INPUTS` | `False` | Sort input files alphabetically |
| `MAX_FRAMES` | `0` | `0`: all frames; `>0`: only first N frames (for quick checks) |

### EXIF, Camera & Altitude

| Parameter | Default | Description |
|-----------|-----------|----------|
| `DEFAULT_FOCAL_LENGTH_MM` | `8.8` | Default focal length if not in EXIF (mm) |
| `DEFAULT_SENSOR_WIDTH_MM` | `13.2` | Default sensor width for unknown cameras (mm) |
| `CAMERA_SENSOR_BY_MODEL` | `{"L1D-20c": 13.2, "FC2204": 6.17}` | Per-EXIF-camera-model sensor width overrides (Mavic 2 Pro / Mavic 2 Zoom); unknown models fall back to `DEFAULT_SENSOR_WIDTH_MM` |
| `YAW_OFFSET_DEG` | `0.0` | Calibration offset applied to EXIF yaw value (degrees) |
| `USE_ROUTE_PROFILES` | `True` | Apply verified profile differences for known routes — see [Per-Route Profile Overrides](#per-route-profile-overrides) |
| `USE_GPS_ALT_REF_SIGN` | `False` | Apply GPS altitude reference sign |
| `RAKIM_DUZELTME` | `26` | DEM datum offset (meters) added when resolving flight altitude — the `+ Correction` term in the GSD formula |
| `BASARI_ESIGI_KM` | `0.07` | Distance (km) below which a prediction is classified as correct (70 m) |
| `MAP_RES_CM_PER_PX` | `29.85` | Reference map resolution (cm/pixel). Used for scale conversion and to turn optical-flow pixel displacement into real-world speed |
| `KENAR_SINIR_PX` | `272` | Positions this close to the map edge are skipped (pixels) |

### Trajectory & On-Screen UI

| Parameter | Default | Description |
|-----------|-----------|----------|
| `UI_BUTTONS_ENABLED` | `True` | On-screen toggle buttons and mouse callback |
| `UI_BUTTON_FONT_SCALE` | `1.0` | Button text size |
| `UI_BUTTON_THICKNESS` | `2` | Button text/frame thickness |
| `UI_BUTTON_SCALE` | `0.5` | Overall scale of the whole UI panel |
| `UI_WINDOW_WIDTH` | `1000` | Width of the `konum` window |
| `UI_WINDOW_HEIGHT` | `1000` | Height of the `konum` window |
| `SHOW_INNER_FRAME` | `False` | Inner frame visibility (initial state) |
| `SHOW_ROI_FRAME` | `True` | ROI frame visibility (initial state) |
| `SHOW_TM_BOXES` | `True` | Template matching boxes visibility (initial state) |
| `DRAW_TRAJECTORY` | `True` | If `False`, the trajectory is never drawn |
| `TRAJECTORY_DRAW_POINTS` | `True` | Also draw a point marker at each step |
| `TRAJECTORY_MAX_POINTS` | `0` | `0`: unlimited; `>0`: keep only the last N points |
| `TRAJECTORY_LINE_THICKNESS` | `15` | Trajectory line thickness |
| `TRAJECTORY_POINT_RADIUS` | `20` | Trajectory point marker radius |

### Optical Flow Speed Channel

| Parameter | Default | Description |
|-----------|-----------|----------|
| `OPTICAL_FLOW_SPEED_ENABLED` | `True` | Enables the optical-flow speed channel, independent of position estimation |
| `OPTICAL_FLOW_MAX_DIMENSION` | `960` | Long-edge cap (pixels) for the frame before tracking, to bound processing cost |
| `OPTICAL_FLOW_MAX_CORNERS` | `800` | Maximum number of Lucas–Kanade corner features to track |
| `OPTICAL_FLOW_MIN_TRACKS` | `20` | Minimum number of surviving tracks required to trust the estimate |
| `OPTICAL_FLOW_MIN_INLIER_RATIO` | `0.45` | Minimum RANSAC inlier ratio required to trust the estimate |
| `OPTICAL_FLOW_RANSAC_THRESHOLD_PX` | `2.5` | RANSAC reprojection threshold (pixels) for the similarity fit |

### Kalman Filter (Position Tracking)

| Parameter | Default | Description |
|-----------|-----------|----------|
| `USE_KALMAN` | `True` | Filters position estimation with Kalman; active only in tracking mode when `KALMAN_IN_BENCHMARK=False` |
| `KALMAN_PROCESS_NOISE` | `30.0` | Process noise std (px). As it gets larger, it adapts to measurements faster (less smoothing) |
| `KALMAN_MEASUREMENT_NOISE` | `12.0` | Measurement noise std (px) — see the empirical note in [Research Tooling](#research-tooling) for why this value was chosen |
| `KALMAN_CONF_GOOD` | `1.0` | 3-way intersection confidence (measurement noise is divided by this; `1.0` = full confidence) |
| `KALMAN_CONF_OK` | `0.5` | 2-way intersection confidence (lower -> less trust in measurement) |
| `KALMAN_WINDOW_FOLLOWS` | `True` | `True`: search frame focuses on (filtered) Kalman position; tracks coasted good position in frames with no intersection -> recovers from outlier clusters |
| `KALMAN_REACQUIRE_FRAMES` | `1` | Filter is reseeded at this many distant and high-confidence measurements |
| `KALMAN_REACQUIRE_JUMP_PX` | `700` | If the measurement is this far from the filter, it is considered a candidate for reacquisition |
| `KALMAN_STEP_GATE_MULT` | `8.0` | Scales the adaptive motion gate with the median step of recent frames |
| `KALMAN_MAX_STEP_PX` | `700` | Base of the adaptive gate; absolute limit when adaptive is off |
| `KALMAN_USE_MOTION` | `True` | Feeds motion derived from accepted measurements into the prediction |
| `KALMAN_MOTION_EMA` | `0.2` | EMA smoothing coefficient for motion speed |
| `KALMAN_MOTION_COAST_DECAY` | `0.7` | Decay rate of motion in coast frames |
| `KALMAN_LOST_GROWTH_PX` | `800` | Gradual growth step of the search window (px/frame) in case of loss (continuous coast), limited by `CERCEVE_BOYUTU_MAX`. Not used when `KALMAN_COV_GATE` is on |
| `KALMAN_COV_GATE` | `False` | **Covariance-based principled mode.** When on, replaces ad-hoc gates with one derived from filter covariance |
| `KALMAN_GATE_SIGMA` | `3.0` | (COV mode) Innovation gate sigma |
| `KALMAN_ROI_SIGMA` | `4.0` | (COV mode) Search window half-width sigma |
| `KALMAN_COV_MOTION_FRAC` | `0.4` | (COV mode) Reduces q to this multiple of median step when `USE_MOTION` is on |
| `KALMAN_GAIN_MAX` | `0.35` | Maximum Kalman gain applied towards the measurement in a single update |
| `KALMAN_OUTPUT_WARMUP_FRAMES` | `0` | `>0`: for the first N Kalman frames the internal state warms up, but the *reported* output is still the raw measurement |
| `KALMAN_WARMUP_RESEED` | `False` | `True`: during warmup, re-seed the state to the raw measurement every frame |
| `KALMAN_IN_BENCHMARK` | `False` | `True`: also apply the Kalman output filter while `BENCHMARK=True` |
| `KALMAN_RAW_ON_UPDATE` | `True` | `True`: report the raw measurement on accepted updates, and the Kalman output only on coast/outlier frames |
| `KALMAN_LOST_SCORE` | `0.0` | If `>0`, TM score below this means "lost" frame. **Tried on this dataset and it didn't help** — good and bad frames land in the same ~0.15–0.25 score band, so `max_val2` isn't a reliable quality signal here. Left at `0`; don't re-try without new evidence. |
| `USE_GPS_REVERT` | `False` | Old recovery using true GPS error; must remain off in GPS-denied experiments |

> Kalman parameters were tuned against this dataset's multi-route experiments; recalibrate for a different platform, frame rate, or map resolution.
>
> **Multi-route results (4 Ürgüp routes, all GPS-denied; RMSE, m):**
>
> | Route | Kalman ON | Visual-only (no KF, revert off) | OFF (GPS-revert crutch) |
> |---|---|---|---|
> | 1_tezde_5 | **37** | 180 | 59 |
> | 3_tezde_7 | **231** | 329 | 202 |
> | 2_tezde_6 | **90** | 2926 | 88 |
> | 6_tezde_4 | **295** | 694 | 205 |
>
> Without using GPS, the Kalman filter absorbs coarse mismatch clusters by coasting, and recovers from lock-in via reacquisition. It matches or beats the GPS-crutched OFF baseline on two routes and stays close on the harder ones — without needing GPS.
>
> **Broader test (8 routes total):** Kalman clearly helps on 5 routes, is neutral on 2 (negligible extra cost on an easy route), and is catastrophic on none. Two routes (`4_tezde_8`, `guz4_tezde_3`) fail intrinsically (~0–50% accuracy): the visual match itself collapses (likely map coverage / model mismatch) — a data problem Kalman cannot fix. See the high-altitude domain-shift note in [Limitations and Notes](#limitations-and-notes).
>
> **Provenance note:** these numbers are exploratory project records. Do not cite them as a published baseline result without matching them to a full configuration, source commit, data split, frame accounting, and asset fingerprints — see [Experiment Protocol and Reproducibility](#experiment-protocol-and-reproducibility).

### Localization Quality, Sensor Fusion & Diagnostics

Parameters enabling functions from the `gps_denied_autonomy.py` module. **All flags default to `False`; WHEN OFF, existing behavior (including Kalman) is EXACTLY preserved.**

Composite localization quality (`USE_QUALITY`) -- produces a continuous confidence [0,1] from the three templates.

| Parameter | Default | Description |
|-----------|-----------|----------|
| `USE_QUALITY` | `False` | `True`: composite confidence is calculated and fed to the Kalman gate |
| `QUALITY_SCORE_THRESHOLD` | `0.35` | BASE threshold for normalized score |
| `QUALITY_CONFIDENCE_THRESHOLD` | `0.40` | Composite confidence threshold |
| `QUALITY_SPREAD_THRESHOLD_PX` | `120.0` | Threshold for three box center spread (px) |
| `NO_INTERSECTION_USE_SEARCH_CENTER` | `True` | If no intersection exists, fall back to the search-window center instead of a single/weak template box |
| `LOW_SCORE_USE_SEARCH_CENTER` | `0.0` | `>0`: if the TM score falls below this threshold, use the search center instead of the intersection |

Sensor fusion (`USE_FUSION`) -- blends raw measurement with previous **output** position based on confidence. Only active when Kalman is OFF.

| Parameter | Default | Description |
|-----------|-----------|----------|
| `USE_FUSION` | `False` | `True`: raw measurement is blended with the prior output position |
| `FUSION_BLEND_GAIN` | `0.75` | Blend gain |
| `FUSION_MAX_JUMP_PX` | `600.0` | Rejected if measurement is further than this * 1.75 from prior |

Diagnostics (`DIAGNOSTIC_ENABLED` / `LOG_QUALITY_CSV`) -- writes triptych PNGs and frame-level metrics.

| Parameter | Default | Description |
|-----------|-----------|----------|
| `DIAGNOSTIC_ENABLED` | `False` | `True`: frame-based triptych PNG + meta JSON + summary.json |
| `DIAGNOSTIC_OUTPUT_DIR` | `"diagnostics"` | Output root folder |
| `LOG_QUALITY_CSV` | `False` | `True`: frame-based quality metrics are written to `tani_kalite_<timestamp>.csv` |

### Per-Route Profile Overrides

When `USE_ROUTE_PROFILES=True`, the pipeline looks up the basename of `ANLIK_DIR` in the `ROUTE_PROFILE_OVERRIDES` dictionary (defined right after `RUN_CFG`) and applies any matching per-route `RUN_CFG` overrides. Today it contains three empirically-derived entries:

| Route | Override | Reason |
|---|---|---|
| `4_tezde_ucus_8` | `YAW_OFFSET_DEG = -8.5` | This route's EXIF yaw needed a fixed calibration offset (found via sweep, 2026-06-11). |
| `3_tezde_ucus_7` | `KALMAN_WINDOW_FOLLOWS = False` | Letting the search window follow the Kalman prediction reduced accuracy on this route; using it as an output-only filter/hold worked better. |
| `5_tezde_ucus_9` | `KALMAN_WINDOW_FOLLOWS = False` | Same reasoning as above. |

Adding a new route does not automatically get a profile — if it needs a similar calibration fix, add its own entry.

### Miscellaneous / Runtime

| Parameter | Default | Description |
|-----------|-----------|----------|
| `WAIT_PER_MODEL` | `False` | Pause after each model |
| `WAIT_ON_EXIT` | `False` | Pause at the end of the program |
| `LOG_LEVEL` | `"WARNING"` | Optional logger level (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Default is `WARNING` so it produces no extra output during normal operation |
| `LOG_TO_FILE` | `False` | If `True`, logs are also written to `tm_run.log` |

---

## Installation

### 1. Python Environment (recommended: Conda)

GDAL/rasterio are unreliable to `pip install` on Windows; conda-forge is far more reliable and is what this project's other tooling (the headless runner) already assumes. Using one shared environment name, `visual_navigation`, keeps things consistent with [headless/README.md](headless/README.md) and with the test suite:

```bash
conda create -n visual_navigation -c conda-forge python=3.11 \
    gdal=3.8.4 rasterio=1.3.10 numpy=1.26.4 pandas=2.2.2 \
    pyproj=3.6.1 affine=2.4.0 pillow=10.4.0 piexif
conda activate visual_navigation

pip install opencv-python==4.10.0.84   # GUI build; use opencv-python-headless on a headless server
pip install tensorflow==2.16.1          # CPU-only — see step 3 for GPU
```

A plain `venv` + pip also works if prebuilt GDAL wheels are available for your platform:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. GDAL Installation (Windows)

GDAL installation on Windows might require extra steps. If `pip install GDAL` fails, use conda:

```bash
conda install -c conda-forge gdal rasterio
```

### 3. CUDA GPU Support (Optional)

To use CUDA acceleration:
- Install NVIDIA GPU drivers.
- Compile OpenCV with CUDA support or use the CUDA build of the `opencv-contrib-python` package.
- Install the appropriate CUDA Toolkit and cuDNN for TensorFlow GPU support.

### 4. Optional Extras

- **Qt control panel**: `pip install PyQt5` — required only for `konum_ui_qt.py`, not listed in `requirements.txt`.
- **Headless server**: `pip install opencv-python-headless` instead of `opencv-python`, and see [headless/README.md](headless/README.md) / `headless/requirements-linux.txt` for the full server setup.

### 5. Data Preparation

- **Maps**: Place georeferenced orthophoto maps (GeoTIFF) in the `haritalar/` folder.
- **Models**: Place trained Keras model files (`.h5`) in the `model/` folder.
- **Images**: Place drone images (JPEG with EXIF data) in the folder pointed to by `ANLIK_DIR` (default `guzergahlar/1_tezde_ucus_5`).
- **DEM**: Place the Digital Elevation Model GeoTIFF file at the path in `DEM_PATH` (default: project root).

---

## Dependencies

| Package | Purpose |
|-------|---------------|
| `tensorflow` / `keras` | Deep learning model inference |
| `opencv-python` (`cv2`) | Image processing, template matching, visualization |
| `rasterio` | GeoTIFF raster file reading, CRS transformations |
| `gdal` (`osgeo`) | DEM (Digital Elevation Model) reading |
| `numpy` | Numerical calculations |
| `pandas` | Writing results in table format |
| `Pillow` (`PIL`) | EXIF data reading, image size control |
| `piexif` | EXIF metadata processing |
| `pyproj` | Coordinate reference system transformations (WGS84 <-> UTM) |
| `affine` | Affine transformation matrix operations |
| `concurrent.futures` | Parallel template matching (Python standard library) |
| `multiprocessing` | Parallel processing support (Python standard library) |
| `PyQt5` *(optional)* | Only for the `konum_ui_qt.py` control-panel front-end — not in `requirements.txt` |

---

## Usage

### Basic Execution

```bash
python template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py
```

### Benchmark Mode
Set the `BENCHMARK` value to `True` in `RUN_CFG` to enable benchmark mode.

### Debug Mode
Set the `DEBUG` value to `True` in `RUN_CFG` to display intermediate images on the screen.

### Alternative Front-Ends

- **Qt control panel** — `python konum_ui_qt.py` runs the same pipeline behind a PyQt5 window with a background worker thread, live metric cards, and the same visibility toggles as the OpenCV HUD. Requires `PyQt5` (`pip install PyQt5`).
- **Headless / server mode** — `python headless/run_headless.py` runs the identical pipeline with every OpenCV window suppressed, for a Linux server with no display; periodically saves annotated PNG frames instead. See [headless/README.md](headless/README.md) for setup, a `systemd` service example, and troubleshooting.

### Reproducible Experiment Runs

`run_localization.py` is a CLI wrapper around the main pipeline that locks a run to one of two named, reproducible modes instead of hand-editing `RUN_CFG`:

```bash
python run_localization.py --mode benchmark --max-frames 50
python run_localization.py --mode simulation --max-frames 50 --sha256-assets
```

| Mode | Sets | Intent |
|---|---|---|
| `benchmark` | `BENCHMARK=True`, `STRICT_GPS_DENIED_INFERENCE=False` | Oracle/legacy comparison — the `BENCHMARK=True` row in [Academic Framework](#academic-framework). |
| `simulation` | `BENCHMARK=False`, `STRICT_GPS_DENIED_INFERENCE=True`, `ALTITUDE_SOURCE=exif_altitude_proxy` | Intended contract for a stricter GPS-denied evaluation — see below. |

Both modes set `WRITE_RUN_ARTIFACTS=True` and pass the resolved config through the `TM_RUN_CFG_JSON` environment variable, which the main script reads and merges into `RUN_CFG`.

> **Current status:** the main pipeline only reacts to the subset of injected keys it already understands (`BENCHMARK`, `MAX_FRAMES`, …). `STRICT_GPS_DENIED_INFERENCE`, `ALTITUDE_SOURCE`, and `WRITE_RUN_ARTIFACTS` are consumed by this CLI and by `localization_policy.py`'s policy functions, but the 4098-line main script does not yet branch on them. Treat `run_localization.py` as a specified, unit-tested contract for the stricter protocol, not as something that changes the reported numbers today — see [Experiment Protocol and Reproducibility](#experiment-protocol-and-reproducibility).

### Runtime Controls
In the `konum` (position) window, visibility settings can be changed with keyboard shortcuts:
- `T`: Toggle Trajectory
- `I`: Toggle Inner Frame
- `O`: Toggle ROI Frame
- `R`: Toggle TM Boxes
- `H`: Collapse/Expand UI panel

### Logging
The script has an optional `logging` infrastructure (`tm` logger). Default level is `WARNING`. Configure via `RUN_CFG`: `LOG_LEVEL` and `LOG_TO_FILE`.

### Tests

```bash
python -m unittest discover -s tests -v
```

Eight suites, split by dependency weight:

| Needs only the standard library | Needs `numpy` / `opencv` / the main module |
|---|---|
| `test_quality.py` (imports `gps_denied_autonomy`) | `test_core_functions.py` (loads the main script) |
| `test_experiment_tracking.py` | `test_kalman.py` (loads the main script) |
| `test_localization_policy.py` | `test_optical_flow_speed.py` (`cv2`, `numpy`) |
| `test_run_localization.py` | `test_tracking_filter.py` (`numpy`) |

`.github/workflows/unit-tests.yml` runs only the left column, on a bare `actions/setup-python` runner with no install step — none of those four import a third-party package. The right column needs the full `visual_navigation` environment from [Installation](#installation): those tests load the main script dynamically via `importlib.util` and are wrapped in `unittest.skipIf`, so a missing dependency (cv2/osgeo/tensorflow) skips them cleanly instead of failing.

---

## Technical Details

### Reading EXIF Data
The `parse_exif()` function reads EXIF data using the PIL library. It extracts Yaw, GPS, Altitude, Focal Length, Camera Model, and Timestamp.

### DEM-Based Multi-Scale Approach
To account for uneven terrain, patches are generated at three different scales corresponding to the Center, Top-Left, and Bottom-Right elevations derived from the DEM.

### Image Preprocessing
1. **Yaw Correction**: Aligns the image North.
2. **Crop Inner Rectangle**: Removes black corners created during rotation.
3. **Scaling**: Scales to match reference map resolution based on GSD.
4. **Patch Extraction**: Extracts 544x544 pixel patches.

### Keras Model Inference
Converts the patches to grayscale, equalizes histograms, normalizes to [-1, 1], and passes them through the Keras model in a single batch to extract feature maps.

### Template Matching
Uses `cv2.TM_CCOEFF_NORMED` to find the highest correlation point on the reference map. 

### Position Determination (Intersection Method)
Calculates the intersection of the matching rectangles from the 3 templates. The center of this intersection area is accepted as the final position estimate.

### Pyramid Search
When `USE_PYRAMID = True`, a two-stage search (coarse-to-fine) is used to speed up the process on large maps.

### CUDA GPU Acceleration
The system automatically checks for CUDA GPU presence and transparently uses it for Template Matching, Image Resizing, and Image Rotation if available.

### Optical Flow Speed Estimation
Independently of the position pipeline, `optical_flow_speed.py` tracks sparse Lucas–Kanade features between consecutive drone frames, filters them with a RANSAC similarity fit, and converts the inlier median displacement into a real-world speed using `MAP_RES_CM_PER_PX` and the EXIF timestamp delta. Enabled by default (`OPTICAL_FLOW_SPEED_ENABLED`); tunable via the `OPTICAL_FLOW_*` parameters (frame downscale, corner count, minimum tracks/inlier ratio, RANSAC threshold).

### Kalman Filter (Position Tracking)
When `USE_KALMAN=True`, the raw intersection center is smoothed using a Constant-Position Kalman filter (state = `(x, y)` only, in map-pixel space) instead of the more common constant-velocity design. This was a deliberate choice, not the default: an earlier constant-*velocity* model plus innovation gating plus window-following **diverged** on this dataset — the filter would coast in the wrong direction, drag the search window with it, corrupt subsequent measurements, and feed the error back into itself (route-wide RMSE went from 59 m to 104 m). The constant-position model never extrapolates a velocity forward — every valid measurement pulls it back — so it cannot run away and lock onto the wrong region the same way. It handles noisy measurements and incorporates an adaptive window, motion predictions, and coasting, all without requiring GPS data. See the empirical results table in [Configuration (RUN_CFG)](#configuration-run_cfg) for route-level numbers.

---

## Coordinate Transformations

| Transformation | Function | Description |
|---|---|---|
| WGS84 -> Pixel | `piksel_bul()` / `piksel_bul_fast()` | Converts a GPS coordinate to a map pixel position |
| Pixel -> WGS84 | `koordinat_bul()` / `make_rc_to_ll()` | Converts a pixel position to a geographic coordinate |
| WGS84 -> UTM | `latlon_to_utm()` | Converts latitude/longitude to a UTM coordinate |
| Haversine | `haversine_distance()` | Great-circle distance between two geographic coordinates |
| Quick Distance | `quick_distance()` | Approximate distance (fast path) |
| Quick Distance UTM | `quick_distance_utm()` | UTM-based distance calculation |

**CRS transforms**: `pyproj.Transformer` converts between EPSG:4326 (WGS84) and the map's/DEM's local CRS.

---

## Evaluation Metrics
The main run reports the following to `sonuclar.csv` / `sonuclar.txt` / `modele_gore_sonuclar.txt`:

**Distance metrics** — RMSE (root mean square error), MAE (mean absolute error), and standard deviation of the error distribution.

**Classification metrics** (threshold: 70 m) — a prediction is scored against whether it falls within `BASARI_ESIGI_KM` and whether a template intersection was found at all:
- **True Positive (TP)** — correct position found and an intersection exists
- **False Positive (FP)** — wrong position found but an intersection exists
- **True Negative (TN)** — wrong position and no intersection
- **False Negative (FN)** — correct position but no intersection

**Derived metrics** — Precision = TP / (TP + FP); Recall = TP / (TP + FN); F-score = 2 × (Precision × Recall) / (Precision + Recall); accuracy % = correct predictions / total × 100.

When run through `run_localization.py`, `experiment_tracking.FrameStatusRecorder.summary()` additionally derives coverage (`accepted / attempted`), p50/p95/max/MAE/RMSE over only the *accepted* frames, and dropout statistics (longest rejected streak, recovery-event count and length) — see [Experiment Protocol and Reproducibility](#experiment-protocol-and-reproducibility).

---

## Experiment Protocol and Reproducibility

For academic runs, a single accuracy or RMSE number is not enough — the same route can look very different depending on configuration, code version, and which frames were even attempted. Two layers exist for this.

**Available today, always applicable:**
- Preserve the git commit ID and working-tree cleanliness alongside every reported number.
- Record the resolved `RUN_CFG` (after any route-profile overrides) and the installed package versions.
- `experiment_tracking.py` provides the building blocks for this: `environment_snapshot()`, `git_revision()` / `git_is_dirty()`, `file_fingerprint()` (metadata or SHA-256 asset hashing), and `build_run_manifest()`, plus a `FrameStatusRecorder` that records exactly one terminal status per frame (`accepted` / `rejected_hold` / `skipped` / `failed`) and derives coverage, dropout-streak, and recovery-time statistics from it. It has no image/GIS dependencies, so a manifest can still be written even if the main run crashes partway through.
- `run_artifacts/<run_id>/` (local, git-ignored) already holds examples of this output: `run_manifest.json` (schema, environment, asset fingerprints, resolved config) and `frame_status_m0.csv` / `frame_summary_m0.json` per model.

**Specified and unit-tested, not yet wired into the main pipeline** (see [Code Map](#code-map)):
- `run_localization.py --mode simulation` is meant to enforce a strict GPS-denied protocol: no true GPS position for the search center, motion prior, DEM query, or recovery after `t=0`; only altitude is still trusted, and only through an explicit, declared source (`select_inference_altitude()` in `localization_policy.py`: `initial_hold`, `exif_altitude_proxy`, or `external_csv` — never a silent fallback to the GPS fix).
- `localization_policy.validate_inference_policy()` raises if an incompatible combination is requested (e.g. `USE_EXIF_MOTION_SEARCH_PRIOR` or `USE_GPS_REVERT` turned on inside strict mode).
- These rules are covered by `tests/test_localization_policy.py` and `tests/test_run_localization.py`, but the main script does not currently import `localization_policy` or branch on `STRICT_GPS_DENIED_INFERENCE`. Until that integration lands, **only `BENCHMARK=False` (the existing adaptive-tracking mode) reflects real, running GPS-denied behavior** — treat `simulation` mode as a documented target, not a currently-enforced guarantee.

When citing numbers from this project (in a thesis, paper, or report), state explicitly which of the two you mean.

---

## Research Tooling

`tools/kalman_sweep.py` runs the main script as a subprocess once per (route × config) combination — overriding `RUN_CFG` the same way `run_localization.py` does, via `TM_RUN_CFG_JSON` — and collects each run's results under `results/kalman_sweep/<run_id>/`. `tools/analyze_kalman_sweep.py` then aggregates those CSVs and ranks candidate parameter sets against a baseline (`off_visual_no_revert`).

> **Empirical note** from a 2026-06-07 multi-route sweep: switching the Kalman output to "output-only smoothing" reduced mean RMSE from **1073 m to 585 m** across the 11 routes that produced valid results in that sweep. This is what motivated the current `KALMAN_MEASUREMENT_NOISE` default (see the inline comment above that key in `RUN_CFG`). Treat this as a local tuning result, not a controlled thesis experiment — re-run the sweep before citing it.

---

## Visualization
Multiple OpenCV windows are used.

### Main map window ("konum")
- Zoomed-in view of the reference map
- **Red box**: Template 1 (top-left scale) match position
- **Green box**: Template 2 (center scale) match position
- **Blue box**: Template 3 (bottom-right scale) match position
- **Black box**: search frame bounds
- **Orange box**: ROI frame (toggle)
- **Yellow dot**: predicted position
- **Green dot**: true GPS position
- **Aircraft icon**: true position and heading
- **Yellow trail / Green trail**: predicted/true trajectory (toggle)
- **Light-blue arrow**: computed speed vector

### HUD (Head-Up Display)
- **HDG**: yaw/heading angle (degrees)
- **ALT**: flight altitude (meters)
- **ERR**: position error (meters)
- **SPD**: speed (`m/s` and `km/h`)
- **Scale bar**: 100-meter reference bar
- **Crosshair**: screen center

The Qt front-end (`konum_ui_qt.py`) renders the same information inside a PyQt5 window with additional live metric cards.

> Note: the speed vector is computed and drawn as an arrow on the map; its components are not broken out separately in the HUD text.

### Crop vs. model window
- Top: cropped and rotated drone image (color)
- Bottom: Keras model output (grayscale)

---

## Output Files
- `sonuclar.csv` / `sonuclar.txt` — latest run, detailed per-frame results.
- `modele_gore_sonuclar.txt` — latest run, summary metrics per model.
- `run_artifacts/<run_id>/` (git-ignored) — `run_manifest.json` plus per-model `frame_status_m*.csv` / `frame_summary_m*.json`, written when `WRITE_RUN_ARTIFACTS` is set (currently only via `run_localization.py`).
- `diagnostics/diag_<timestamp>_m<model_index>/` (git-ignored) — per-frame triptych PNGs (crop | model output | matched reference region) + meta JSON, plus a run-end `summary.json`, when `DIAGNOSTIC_ENABLED=True`.
- `tani_kalite_<timestamp>.csv` — per-frame quality metrics when `LOG_QUALITY_CSV=True`.
- `results/` (git-ignored) — archived manual/sweep runs, including `results/kalman_sweep/`.
- `tm_run.log` — optional logger output file when `LOG_TO_FILE=True`.

---

## Camera Support
Per-EXIF-camera-model sensor widths are declared in `CAMERA_SENSOR_BY_MODEL`: `L1D-20c` → 13.2 mm (DJI Mavic 2 Pro), `FC2204` → 6.17 mm (DJI Mavic 2 Zoom). Any other EXIF camera model falls back to `DEFAULT_SENSOR_WIDTH_MM` (13.2 mm); `DEFAULT_FOCAL_LENGTH_MM` (8.8 mm) is used when EXIF has no focal length at all.

---

## Limitations and Notes
- Map and model counts in their respective folders must match (one Keras model is paired with one reference map per iteration).
- Images must contain GPS, altitude, and focal length in EXIF (MakerNote `FlightDegree` for yaw).
- Query positions must fall within the DEM extent.
- Large maps/DEMs (multi-GB GeoTIFFs) require significant RAM.
- A 70 m threshold (`BASARI_ESIGI_KM=0.07`) is used to classify a prediction as correct in the aggregate metrics.
- **Known high-altitude failure mode:** routes flown at roughly ≥1000 m AGL (e.g. `4_tezde_ucus_8`) currently perform poorly. This has been characterized as a **model domain-shift issue** — the Keras feature-extraction model was not trained on imagery at that altitude band — rather than a search-coverage or scale-calculation bug; recalibrating `YAW_OFFSET_DEG` (see [Per-Route Profile Overrides](#per-route-profile-overrides)) does not fix it. Retraining/fine-tuning on higher-altitude imagery is the likely fix, not a `RUN_CFG` change.
- `ROUTE_PROFILE_OVERRIDES` currently hard-codes calibration fixes for three specific routes; a new route may need its own entry.
- Default parameters are tuned for ~30 cm/px imagery around Ürgüp; expect to recalibrate `MAP_RES_CM_PER_PX`, `DEFAULT_SENSOR_WIDTH_MM` / `CAMERA_SENSOR_BY_MODEL`, and `RAKIM_DUZELTME` for a different sensor, region, or altitude band.
- See [Experiment Protocol and Reproducibility](#experiment-protocol-and-reproducibility) for the gap between the currently-running GPS-denied behavior and the stricter, specified-but-not-yet-wired `simulation` protocol.

---

## Citation
There is no published `CITATION.cff` or DOI for this repository yet. When the relevant thesis/article is published, citation should include author, title, institution, year, version, and permalink.

---

## License
This project was developed within the scope of a thesis study at Cappadocia University. There is no separate `LICENSE` file; accessibility to the source code does not automatically grant permission for redistribution or derivative works. Contact the project owners for usage and licensing.
