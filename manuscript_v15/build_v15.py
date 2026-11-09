#!/usr/bin/env python3
"""Build Arik_Emrahoglu_UnmannedSystems_v15.docx from the submitted v14 file.
Part 1 (this file): structural and methods edits. Part 2 (results_v15.py): results-dependent text and tables."""
import sys, docx, copy
from docx_tools import *
import text_methods as T
REJ_G, REJ_B = '1.8%', '8.3%'  # texture survey on the 500 m grid (texture_survey.csv)

import os
# the submitted (v14) manuscript; set V15_SRC or pass --src to make_v15.py
SRC = os.environ.get('V15_SRC', r'C:\Users\ertugrul\OneDrive - kapadokya.edu.tr\Arastirma\Tezden_Makale\Unmanned_Systems\Arik_Emrahoglu_UnmannedSystems.docx')
if not os.path.exists(SRC):
    SRC = '/mnt/user-data/uploads/Tezden_Makale/Unmanned_Systems/Arik_Emrahoglu_UnmannedSystems.docx'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'v15_stage1.docx'
d = docx.Document(SRC)

# ---------- references: insert Schleiss as no. 20, shift the rest ----------
n = renumber_citations(d, 20)
print('citations bumped:', n)
i19, p19 = find_par(d, '19. J. Kinnari')
for i in range(i19 + 1, len(d.paragraphs)):
    p = d.paragraphs[i]
    m = re.match(r'^(\d+)\. ', p.text)
    if m and p.runs:
        num = int(m.group(1))
        if num >= 20:
            r0 = p.runs[0]
            r0.text = r0.text.replace(f'{num}. ', f'{num + 1}. ', 1)
insert_par_after(p19, '20. M. Schleiss, Translating aerial images into street-map-like representations for visual self-localization of UAVs, Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci. XLII-2/W13 (2019) 575-580.')

# ---------- title and keywords ----------
set_par(d.paragraphs[0], T.TITLE)
i, p = find_par(d, 'Keywords:')
set_par(p, T.KEYWORDS)

# ---------- introduction ----------
i, p = find_par(d, 'An intermediate strategy keeps the simple correlation matcher')
set_par(p, "An intermediate strategy keeps the simple correlation matcher and instead learns a transformation that moves both the reference and the query into a common domain in which correlation is reliable. Fragoso et al. trained a seasonally invariant deep transform for visual terrain-relative navigation;^{17} Bianchi and Barfoot localized a UAV against autoencoded satellite images;^{18} Kinnari et al. learned dense season-invariant features for matching against planar maps;^{19} Schleiss translated aerial images into map-like renderings learned from OpenStreetMap and matched them to the map by template matching;^{20} and Mughal et al., Hou et al. and Jaimes et al. used contextual, deep or semantic features to bridge the UAV-satellite gap.^{21-23} Autoencoders are attractive in this role because they can be trained without manual labels, are compact, and their output is an image that any established matcher can consume.^{24,25}")

i, p = find_par(d, 'This article is derived from, and extends, the first author')
set_par(p, T.INTRO_QUESTION)

i, p = find_par(d, 'A flight evaluation over Ürgüp')
set_par(p, T.CONTRIB_FLIGHT)
q = insert_par_after(p, T.CONTRIB_BASELINES)
i, p = find_par(d, 'An explicit statement of validity boundaries')
set_par(p, "An explicit statement of validity boundaries: all quantitative results are in-domain for the Ürgüp region, the eleven doctoral routes were used to select Model 4 in the doctoral study (the benchmark-selected model never saw flight imagery before the replay), GPS metadata provides the reference (not the measurement), and the pipeline is evaluated offline as a candidate position-fix source rather than as an end-to-end autonomous navigation system.")

# ---------- related work ----------
i, p = find_par(d, 'Learned image transforms for correlation-based localization.')
set_par(p, "Learned image transforms for correlation-based localization. Closest to the present work are methods that learn a transform and keep a classical matcher. Fragoso et al.^{17} trained a convolutional neural network to map camera images and orthophotos into a season-invariant domain, preserving the geometry and uncertainty of the classical matcher. Bianchi and Barfoot^{18} compressed satellite images with an autoencoder to accelerate UAV localization. Kinnari et al.^{19} learned dense features robust to seasonal change and fused them with a point-mass filter over planar maps. Mughal et al.^{21} and Hou et al.^{22} used deep contextual features and DEM-aided satellite features, respectively, for UAV pose estimation, and Jaimes et al.^{23} applied unsupervised semantic segmentation to emphasize stable structures before matching.")
insert_par_after(p, T.RELATED_WORK_ADD)

# ---------- 3.1 ----------
i, p = find_par(d, 'Fig. 1 summarizes the pipeline.')
set_par(p, T.SYS_OVERVIEW)
# ---------- 3.4 ----------
i, p = find_par(d, 'Height above ground is required to scale UAV frames.')
set_par(p, T.DEM_PAR)
# ---------- 3.5 ----------
i, p = find_par(d, 'Flight imagery was acquired with a DJI Mavic 2 Pro')
set_par(p, T.CAMERA_PAR)
i, p = find_par(d, 'Each frame is converted to grayscale, rotated to north')
set_par(p, T.ROTATE_PAR)
i, p = find_par(d, 'From the normalized frame three 544')
set_par(p, T.CROPS_PAR)
# ---------- 3.6 ----------
i, p = find_par(d, '3.6. NCC localization')
set_par(p, '3.6. NCC localization, three-crop consensus and tracking')
i, p = find_par(d, 'Here T̄ is the mean of the template')
set_par(p, T.NCC_PAR_2)
insert_par_after(p, T.TRACKING_PAR)
# ---------- 3.7 ----------
i, p = find_par(d, '3.7. Flight test campaign')
set_par(p, '3.7. Flight test campaign and evaluation protocol')
i, p = find_par(d, 'Test flights took place in Ürgüp district')
set_par(p, T.CAMPAIGN_PAR)
i, p = find_par(d, 'A position estimate is counted as correct if it lies within 30 m')
set_par(p, T.PROTOCOL_PAR)
q = insert_par_after(p, T.MODES_PAR.replace('{F48_CKPT}', '1850'))
i, p = find_par(d, 'Before the flight evaluation, four simulation stages')
set_par(p, T.SIM_PAR)
# Fig. 4 caption (consensus description)
i, p = find_par(d, 'Fig. 4. (a) Nadir UAV frames over Ürgüp')
set_par(p, "Fig. 4. (a) Nadir UAV frames over Ürgüp before (left) and after (right) the cross-autoencoder transformation. (b) Three-crop consensus on the structural reference raster: the red, green and blue squares are the NCC candidate windows of the upper-left, central and lower-right crops; once the known crop displacement is removed, their implied frame centers must agree within 48 px for a position to be output (heading and altitude of the frame are shown in the corner). The structural reference raster is derived from Google Maps satellite imagery © Google.")
# Fig. 1 caption
i, p = find_par(d, 'Fig. 1. Processing pipeline.')
set_par(p, "Fig. 1. Processing pipeline. The same structural transform - the cross-autoencoder with identical weights, or the gradient-magnitude baseline - is applied to the reference mosaic (offline) and to the UAV frame crops (online); localization is performed by normalized cross-correlation between the two structural representations followed by a three-crop consistency check, inside a search window that follows the previous fix through a visual-odometry prior.")

# ---------- 3.8 benchmark: query sampling with gate rejection rates ----------
import text_bench as TB
i, p = find_par(d, 'Query sampling. The common extent is divided')
set_par(p, TB.QUERY_SAMPLING.replace('{REJ_G}', REJ_G).replace('{REJ_B}', REJ_B))
i, p = find_par(d, 'Matching and search modes. The 544')
insert_par_after(p, TB.BENCH_SOBEL_METHOD)
# ---------- 4.4: spatial hold-out of the checkpoint selection ----------
i, p = find_par(d, 'Table 7 lists the best checkpoint of each finalist lineage')
txt = p.text
cut = txt.index('Because this final choice uses the district data')
set_par(p, txt[:cut].rstrip())
sel = TB.SELECTION_SPLIT
for k, v in TB.SELECTION_SPLIT_VALUES.items(): sel = sel.replace('{' + k + '}', v)
insert_par_after(p, sel)

d.save(OUT)
print('saved', OUT)
