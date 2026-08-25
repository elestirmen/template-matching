# UAV Image Positioning - Deep Learning Supported Geolocalization via Template Matching

*[Türkçe sürüm için tıklayın (README.tr.md)](README.tr.md)*

This project is an experimental computer vision system that investigates the automated positioning of UAV (Unmanned Aerial Vehicle) images on a **reference orthophoto map**. It combines deep learning-based image transformation with multi-scale template matching, supporting controlled GPS-centric benchmark and sequential visual tracking scenarios as separate operating modes.

> **Research prototype:** The software is intended for academic experiments and method development. It is not a validated product for safety-critical navigation or standalone real flight control. Default parameters are specific to approximately 30 cm/pixel data around Ürgüp; recalibration is required for different regions, altitudes, and sensors.

---

## Table of Contents

- [Overview](#overview)
- [Academic Framework](#academic-framework)
- [System Architecture](#system-architecture)
- [Workflow (Pipeline)](#workflow-pipeline)
- [Folder Structure](#folder-structure)
- [Configuration (RUN\_CFG)](#configuration-run_cfg)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Runtime Controls](#runtime-controls)
- [Technical Details](#technical-details)
  - [Reading EXIF Data](#reading-exif-data)
  - [DEM-Based Multi-Scale Approach](#dem-based-multi-scale-approach)
  - [Image Preprocessing](#image-preprocessing)
  - [Keras Model Inference](#keras-model-inference)
  - [Template Matching](#template-matching)
  - [Position Determination (Intersection Method)](#position-determination-intersection-method)
  - [Pyramid Search](#pyramid-search)
  - [CUDA GPU Acceleration](#cuda-gpu-acceleration)
  - [Kalman Filter (Position Tracking)](#kalman-filter-position-tracking)
- [Coordinate Transformations](#coordinate-transformations)
- [Evaluation Metrics](#evaluation-metrics)
- [Experiment Protocol and Reproducibility](#experiment-protocol-and-reproducibility)
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

---

## Workflow (Pipeline)

When the script is executed, the following steps are performed sequentially:

### 1. Initialization and Resource Loading
- All parameters are read from the `RUN_CFG` configuration dictionary.
- Map files (`haritalar/`), model files (`model/`), and instantaneous images (`parcalar/`) are listed.
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
- `Flight Altitude = GPS Altitude - Terrain Elevation + Correction`
- `GSD (cm/px) = (Sensor Width x Flight Altitude x 100) / (Focal Length x Image Width)`

#### 3.3 Image Preprocessing
- The image is aligned North by rotating it by the **inverse of the yaw angle**.
- The **largest internal rectangle** is cropped from the rotated image (black corners are removed).
- It is scaled according to the GSD ratio to match the reference map resolution.
- 544x544 pixel patches are cropped at three different scales (based on center, top-left, bottom-right elevation).

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
- In normal mode: the search frame for the next frame is narrowed to the region close to the current prediction.
- If matching fails, the frame is expanded.
- In benchmark mode: a fixed frame centered around the EXIF GPS is used for every frame.

### 4. Result Reporting
- RMSE, MAE, standard deviation are calculated.
- Precision, Recall, and F-score are calculated.
- Results are written to the `sonuclar.csv`, `sonuclar.txt`, and `modele_gore_sonuclar.txt` files.

---

## Folder Structure

```text
template matching/
|
+-- template_matching_parallel_processing_560_hizli_solust_sagalt_
|   koordinat_fonksiyonlar_icinde_cursor.py   # Main script
|
+-- haritalar/              # Georeferenced orthophoto map files (.tif)
+-- model/                  # Keras deep learning models (.h5)
+-- parcalar/               # Drone/UAV images to process
|
+-- anlik/                  # Instantaneous image folder (alternative)
+-- anlik_t/                # Instantaneous image folder (alternative)
+-- temp/                   # Temporary files
+-- haritalar_top/          # Additional map files
+-- top_modeller/           # Additional model files
+-- arsiv/                  # Archived files
|
+-- bern sehri template match/  # Bern city test data
|
+-- ana_harita_urgup_30_cm_utm_elevation.tif  # DEM file (default)
|
+-- sonuclar.csv            # Output: detailed results (CSV)
+-- sonuclar.txt            # Output: detailed results (text)
+-- modele_gore_sonuclar.txt # Output: model-based summary metrics
|
+-- README.md               # This file
+-- README.tr.md            # Turkish translation
+-- .gitignore
```

---

## Configuration (RUN_CFG)

All runtime parameters are managed from a single `RUN_CFG` dictionary at the beginning of the file. After this dictionary is read, it is converted into type-safe constants (bool/int/float).

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
| `HARITA_DIR` | `"haritalar"` | Folder containing map files |
| `MODEL_DIR` | `"model"` | Folder containing Keras model files |
| `ANLIK_DIR` | `"guzergahlar/1_tezde_ucus_5"` | Folder containing drone images |
| `DEM_PATH` | `"ana_harita_urgup_30_cm_utm_elevation.tif"` | DEM raster file path |
| `HARITA_DOSYALARI` | `[]` | List of specific map files (empty = all in folder) |
| `MODEL_DOSYALARI` | `[]` | List of specific model files (empty = all in folder) |
| `SORT_INPUTS` | `False` | Sort input files alphabetically |
| `MAX_FRAMES` | `0` | `0`: all frames; `>0`: only first N frames (for quick checks) |
| `DEFAULT_FOCAL_LENGTH_MM` | `8.8` | Default focal length if not in EXIF (mm) |
| `DEFAULT_SENSOR_WIDTH_MM` | `13.2` | Default sensor width for unknown cameras (mm) |
| `YAW_OFFSET_DEG` | `0.0` | Calibration offset applied to EXIF yaw value (degrees) |
| `USE_ROUTE_PROFILES` | `True` | Apply verified profile differences for known routes |
| `USE_GPS_ALT_REF_SIGN` | `False` | Apply GPS altitude reference sign |
| `WAIT_PER_MODEL` | `False` | Pause after each model |
| `WAIT_ON_EXIT` | `False` | Pause at the end of the program |
| `LOG_LEVEL` | `"WARNING"` | Optional logger level (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Default is `WARNING` so it produces no extra output during normal operation |
| `LOG_TO_FILE` | `False` | If `True`, logs are also written to `tm_run.log` |
| `MAP_RES_CM_PER_PX` | `29.85` | Reference map resolution (cm/pixel). Used in scaling and speed calculations |
| `KENAR_SINIR_PX` | `272` | Positions this close to the map edge are skipped (pixels) |
| `OPTICAL_FLOW_SPEED_ENABLED` | `True` | Enables the optical-flow speed channel independent of position estimation |

Kalman filter (position tracking) settings:

| Parameter | Default | Description |
|-----------|-----------|----------|
| `USE_KALMAN` | `True` | Filters position estimation with Kalman; active only in tracking mode when `KALMAN_IN_BENCHMARK=False` |
| `KALMAN_PROCESS_NOISE` | `30.0` | Process noise std (px). As it gets larger, it adapts to measurements faster (less smoothing) |
| `KALMAN_MEASUREMENT_NOISE` | `12.0` | Measurement noise std (px); adapted with confidence value |
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
| `KALMAN_LOST_GROWTH_PX` | `800` | GRADUAL growth step of the search window (px/frame) in case of loss (continuous coast), limited by `CERCEVE_BOYUTU_MAX`. Not used when `KALMAN_COV_GATE` is on |
| `KALMAN_COV_GATE` | `False` | **COVARIANCE-BASED PRINCIPLED MODE.** When on, replaces ad-hoc gates with one derived from filter covariance |
| `KALMAN_GATE_SIGMA` | `3.0` | (COV mode) Innovation gate sigma |
| `KALMAN_ROI_SIGMA` | `4.0` | (COV mode) Search window half-width sigma |
| `KALMAN_COV_MOTION_FRAC` | `0.4` | (COV mode) Reduces q to this multiple of median step when `USE_MOTION` is on |
| `KALMAN_GAIN_MAX` | `0.35` | Maximum Kalman gain applied towards the measurement in a single update |
| `KALMAN_LOST_SCORE` | `0.0` | If `>0`, TM score below this means "lost" frame. Left at `0` for this dataset. |
| `USE_GPS_REVERT` | `False` | Old recovery using true GPS error; must remain off in GPS-denied experiments |

### Localization Quality, Sensor Fusion, and Diagnostics

Parameters enabling functions from the `gps_denied_autonomy.py` module. **All flags default to `False`; WHEN OFF, existing behavior (including Kalman) is EXACTLY preserved.**

Composite localization quality (`USE_QUALITY`) -- produces a continuous confidence [0,1] from the three templates.

| Parameter | Default | Description |
|-----------|-----------|----------|
| `USE_QUALITY` | `False` | `True`: composite confidence is calculated and fed to the Kalman gate |
| `QUALITY_SCORE_THRESHOLD` | `0.35` | BASE threshold for normalized score |
| `QUALITY_CONFIDENCE_THRESHOLD` | `0.40` | Composite confidence threshold |
| `QUALITY_SPREAD_THRESHOLD_PX` | `120.0` | Threshold for three box center spread (px) |

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
| `LOG_QUALITY_CSV` | `False` | `True`: frame-based quality metrics are written to `tani_kalite.csv` |

UI and visibility focused settings:

| Parameter | Default | Description |
|-----------|-----------|----------|
| `UI_BUTTONS_ENABLED` | `True` | On-screen toggle buttons |
| `UI_WINDOW_WIDTH` | `1000` | Width of the `konum` window |
| `UI_WINDOW_HEIGHT` | `1000` | Height of the `konum` window |
| `SHOW_INNER_FRAME` | `False` | Inner frame visibility |
| `SHOW_ROI_FRAME` | `True` | ROI frame visibility |
| `SHOW_TM_BOXES` | `True` | Template matching boxes visibility |

---

## Installation

### 1. Python Environment

Python 3.11 is recommended. For GDAL/rasterio installation on Windows, using Conda is more reliable:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install tensorflow opencv-python rasterio gdal numpy pandas pillow piexif pyproj affine
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

### 4. Data Preparation

- **Maps**: Place georeferenced orthophoto maps (GeoTIFF) in the `haritalar/` folder.
- **Models**: Place trained Keras model files (`.h5`) in the `model/` folder.
- **Images**: Place drone images (JPEG with EXIF data) in the `parcalar/` folder.
- **DEM**: Place the Digital Elevation Model GeoTIFF file in the project root directory.

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
Unit tests for pure helper/math functions are in the `tests/` folder:
```bash
python -m unittest discover -s tests -v
```

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

### Kalman Filter (Position Tracking)
When `USE_KALMAN=True`, the raw intersection center is smoothed using a Constant-Position Kalman filter operating in map-pixel space. This handles noisy measurements and incorporates an adaptive window, motion predictions, and coasting without requiring GPS data.

---

## Coordinate Transformations
The system performs conversions between WGS84, Map Pixel coordinates, and UTM utilizing the `pyproj.Transformer`.

---

## Evaluation Metrics
Calculates RMSE, MAE, Standard Deviation, Precision, Recall, and F-Score for the localization results. Outputs are saved to CSV and text files.

---

## Experiment Protocol and Reproducibility
For academic runs, average accuracy or RMSE is not enough. Commit IDs, configuration, dependencies, and frame-level state accounting should be preserved via the `experiment_tracking.py` utilities.

---

## Visualization
Multiple OpenCV windows are used. The main window ("konum") shows the reference map, bounding boxes for the 3 scales, search frames, trajectory, and a HUD (Head-Up Display) with flight telemetry.

---

## Output Files
- `sonuclar.csv`: Detailed results in CSV format.
- `sonuclar.txt`: Detailed results in text table format.
- `modele_gore_sonuclar.txt`: Summary metrics per model.

---

## Camera Support
Built-in sensor widths for DJI Mavic 2 Pro (13.2 mm) and Mavic 2 Zoom (6.17 mm). Uses a default for unknown cameras.

---

## Limitations and Notes
- Map and Model counts in their respective folders must match.
- Images must contain GPS, altitude, and focal length in EXIF.
- Queries must fall within the DEM extent.
- High RAM consumption for large maps.
- A threshold of 70m is used for classification success.

---

## Citation
There is no published `CITATION.cff` or DOI for this repository yet. When the relevant thesis/article is published, citation should include author, title, institution, year, version, and permalink.

---

## License
This project was developed within the scope of a thesis study at Cappadocia University. There is no separate `LICENSE` file; accessibility to the source code does not automatically grant permission for redistribution or derivative works. Contact the project owners for usage and licensing.
