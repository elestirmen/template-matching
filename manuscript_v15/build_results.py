#!/usr/bin/env python3
"""Stage 2 of v15: results-dependent text and tables.
Usage: python build_results.py TABLES_DIR STAGE1.docx OUT.docx [EXTRA.json]
TABLES_DIR holds make_tables.py outputs (by_group.csv, by_route.csv, by_group_area.csv);
EXTRA.json holds numbers from the other analyses (texture probe, reconstruction control, bias analysis).
"""
import sys, json, os, re, math
import pandas as pd, numpy as np, docx
from docx_tools import *
import results_tables as RT

TD, SRC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EXTRA = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and os.path.exists(sys.argv[4]) else {}
G = pd.read_csv(os.path.join(TD, 'by_group.csv'))
R = pd.read_csv(os.path.join(TD, 'by_route.csv'))
GA = pd.read_csv(os.path.join(TD, 'by_group_area.csv'))
d = docx.Document(SRC)


def gv(run, group, col, nd=1):
    x = G[(G.run == run) & (G.group == group)]
    if x.empty or pd.isna(x.iloc[0][col]): return '?'
    v = float(x.iloc[0][col])
    return f'{v:,.{nd}f}' if nd else f'{int(round(v)):,}'


def gav(run, group, area, col, nd=1):
    x = GA[(GA.run == run) & (GA.group == group) & (GA.area == area)]
    if x.empty or pd.isna(x.iloc[0][col]): return '?'
    v = float(x.iloc[0][col]); return f'{v:,.{nd}f}' if nd else f'{int(round(v)):,}'


def rv(run, route, col, nd=1):
    x = R[(R.run == run) & (R.route == route)]
    if x.empty or pd.isna(x.iloc[0][col]): return '?'
    v = float(x.iloc[0][col]); return f'{v:,.{nd}f}' if nd else f'{int(round(v)):,}'


def rng(run, routes, col, nd=1):
    vals = [float(R[(R.run == run) & (R.route == r)].iloc[0][col]) for r in routes if not R[(R.run == run) & (R.route == r)].empty]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals: return '?'
    return f'{min(vals):.{nd}f}-{max(vals):.{nd}f}'


X = EXTRA.get
URB = ['Urban%d' % i for i in range(1, 9)]; RUR = ['Rural1', 'Rural2', 'Rural3']; PRO_URB = ['Urban4', 'Urban5', 'Urban6', 'Urban7', 'Urban8']

# ------------------------------------------------------------------ 4.1
i, p = find_par(d, '4.1. Simulation-based model selection')
set_par(p, '4.1. Representations compared on the same flight frames')
i, p = find_par(d, 'In Simulation 1 (Bing patches on the transformed Google mosaic')
def fv(run, group, col):
    x = G[(G.run == run) & (G.group == group)]
    return float(x.iloc[0][col]) if len(x) and not pd.isna(x.iloc[0][col]) else float('nan')
best_or = 'f48k4s1_oracle' if fv('f48k4s1_oracle', 'doctoral', 'success_pct') >= fv('m4e300_oracle', 'doctoral', 'success_pct') else 'm4e300_oracle'
d_av = fv(best_or, 'doctoral', 'availability_pct') - fv('gray_oracle', 'doctoral', 'availability_pct')
d_pr = fv(best_or, 'doctoral', 'precision_pct') - fv('gray_oracle', 'doctoral', 'precision_pct')
mech = (f"The gain comes mainly from availability - the share of frames for which the three crops agree - which rises "
        f"from {gv('gray_oracle', 'doctoral', 'availability_pct')}% to {gv(best_or, 'doctoral', 'availability_pct')}%, "
        f"while the precision of accepted fixes changes from {gv('gray_oracle', 'doctoral', 'precision_pct')}% to "
        f"{gv(best_or, 'doctoral', 'precision_pct')}%. " if d_av >= d_pr else
        f"Both the availability (from {gv('gray_oracle', 'doctoral', 'availability_pct')}% to "
        f"{gv(best_or, 'doctoral', 'availability_pct')}%) and the precision of accepted fixes (from "
        f"{gv('gray_oracle', 'doctoral', 'precision_pct')}% to {gv(best_or, 'doctoral', 'precision_pct')}%) increase. ")
c1550 = fv('f48k4s1c1550_oracle', 'doctoral', 'success_pct'); c1850 = fv('f48k4s1_oracle', 'doctoral', 'success_pct')
rel = 'close to' if abs(c1550 - c1850) <= 1.5 else ('above' if c1550 > c1850 else 'below')
set_par(p, (
    "Table 3 pools the replay of all sixteen flights on one set of eligible frames. In the oracle replay every eligible "
    "frame is matched independently inside a 3 × 3 km window around its GPS position, so the comparison isolates the "
    f"representation. On the {gv('gray_oracle', 'doctoral', 'eligible', 0)} eligible frames of the eleven doctoral "
    f"routes, untransformed grayscale NCC places {gv('gray_oracle', 'doctoral', 'success_pct')}% of the frames within "
    f"30 m and the classical gradient-magnitude representation {gv('sobel_oracle', 'doctoral', 'success_pct')}%, "
    f"whereas the cross-autoencoders reach {gv('m4e300_oracle', 'doctoral', 'success_pct')}% (Model 4 lineage, "
    f"stride 2) and {gv('f48k4s1_oracle', 'doctoral', 'success_pct')}% (benchmark-selected f48-k4-s1). " + mech +
    f"The median error of accepted fixes (about {gv(best_or, 'doctoral', 'median_m', 0)} m) is set by the GPS reference "
    "and the camera geometry rather than by the representation (Section 4.2). On the "
    f"{gv('gray_oracle', 'heldout', 'eligible', 0)} eligible frames of the additional flights the four representations "
    f"reach {gv('gray_oracle', 'heldout', 'success_pct')}%, {gv('sobel_oracle', 'heldout', 'success_pct')}%, "
    f"{gv('m4e300_oracle', 'heldout', 'success_pct')}% and {gv('f48k4s1_oracle', 'heldout', 'success_pct')}%. "
    f"Checkpoint 1550 of the f48-k4-s1 run, which was best on the core benchmark mosaics, gives "
    f"{gv('f48k4s1c1550_oracle', 'doctoral', 'success_pct')}% on the doctoral routes, {rel} checkpoint 1850. "
    "For reference, the doctoral study had selected Model 4 with three simulations (Section 3.7); those selection "
    "results are not used as evidence here."))
i, p = find_par(d, 'Table 3. Pooled comparison of the selected network')
set_par(p, "Table 3. Pooled replay results of the sixteen flights (30 m threshold). Oracle: each frame searched independently in a 10,000 px (about 3 × 3 km) window centered on its GPS position. GNSS-aided: window follows the previous fix, frame scaled with its own GNSS altitude and DEM look-up. Causal: GNSS read only at the first frame of a route. Frames = eligible frames, identical in every row (Section 3.7); Succ. = correct fixes / eligible frames; Avail. = accepted fixes / eligible frames; Prec. = correct / accepted fixes; Med. and RMSE over accepted fixes.^{a}")
hdr, rows = RT.pooled_table(G)
tmpl = d.tables[2]
t = build_table(d, tmpl, hdr, rows, [800, 1950, 700, 640, 640, 620, 620, 720, 700, 640, 620, 620],
                group_header=[('', 2), ('Eleven doctoral routes', 6), ('Five additional flights', 4)], left_cols=2)
replace_table(d, d.tables[2], t)
# footnote paragraph after Table 3
tbl3 = d.tables[2]._tbl
nxt = tbl3.getnext()
from docx.text.paragraph import Paragraph
note_anchor = Paragraph(nxt, p._parent) if nxt is not None and nxt.tag.endswith('}p') else None
fn3 = ("^{a} Urban 9-11 and Rural 4-5. All 435 frames of Rural 4 (180 m above take-off) are too small at map resolution "
       "to contain a 544 px crop, and 106 of the 265 frames of Rural 5 lie outside the DEM; in tracking almost all eligible "
       "frames of Rural 5 are lost because the predicted footprint leaves the DEM. The additional-flight block is therefore "
       "dominated by the three urban flights. Gradient magnitude and checkpoint 1550 were replayed in the oracle and causal "
       "modes only.")
if note_anchor is not None and note_anchor.text.strip() == '':
    set_par(note_anchor, fn3)
    for r_ in note_anchor.runs: r_.font.size = docx.shared.Pt(9)
else:
    # insert a small paragraph after the table using the style of the old Table 4 footnote
    _, fpar = find_par(d, 'a Date inferred from the flight log') if any(pp.text.startswith('a Date inferred') for pp in d.paragraphs) else (None, None)
    newp = copy.deepcopy((fpar or p)._p)
    for ch in list(newp):
        if ch.tag != qn('w:pPr'): newp.remove(ch)
    tbl3.addnext(newp)
    set_par(Paragraph(newp, p._parent), fn3, keep_rpr=False)

# ------------------------------------------------------------------ 4.2
i, p = find_par(d, '4.2. Flight tests')
set_par(p, '4.2. Route-level results and GNSS-independent tracking')
i, p = find_par(d, 'All route results in this section were obtained with the thesis-selected Model 4')
ca_m4, ca_f = fv('m4e300_track_causal', 'doctoral', 'success_pct'), fv('f48k4s1_track_causal', 'doctoral', 'success_pct')
or_m4, or_f = fv('m4e300_oracle', 'doctoral', 'success_pct'), fv('f48k4s1_oracle', 'doctoral', 'success_pct')
pairs = [('gray_oracle', 'gray_track_causal'), ('sobel_oracle', 'sobel_track_causal'), ('m4e300_oracle', 'm4e300_track_causal'),
         ('f48k4s1_oracle', 'f48k4s1_track_causal'), ('f48k4s1c1550_oracle', 'f48k4s1c1550_track_causal')]
n_better = sum(fv(c, 'doctoral', 'success_pct') > fv(o, 'doctoral', 'success_pct') for o, c in pairs)
words = {5: 'all five', 4: 'four of the five', 3: 'three of the five', 2: 'two of the five', 1: 'one of the five', 0: 'none of the five'}
n_better_ho = sum(fv(c, 'heldout', 'success_pct') > fv(o, 'heldout', 'success_pct') for o, c in pairs)
trk = (f"On the doctoral routes, tracking with a 600 m window around the predicted position is more reliable than the "
       f"oracle search over 3 × 3 km for {words[n_better]} representations, because the smaller window contains fewer "
       "competing peaks, although frames lost after a tracking failure count as failures; on the additional flights it is "
       f"more reliable for {words[n_better_ho]}, mainly because the track of Rural 5 is lost. ")
set_par(p, (
    trk + "In the causal replay, which reads GNSS only at the first frame of each route, the Model 4 lineage localizes "
    f"{gv('m4e300_track_causal', 'doctoral', 'success_pct')}% of the eligible doctoral frames within 30 m, f48-k4-s1 "
    f"checkpoint 1850 {gv('f48k4s1_track_causal', 'doctoral', 'success_pct')}% and checkpoint 1550 "
    f"{gv('f48k4s1c1550_track_causal', 'doctoral', 'success_pct')}% (precision "
    f"{gv('m4e300_track_causal', 'doctoral', 'precision_pct')}%, {gv('f48k4s1_track_causal', 'doctoral', 'precision_pct')}% "
    f"and {gv('f48k4s1c1550_track_causal', 'doctoral', 'precision_pct')}%; Table 3), against {gv('gray_track_causal', 'doctoral', 'success_pct')}% for grayscale NCC and "
    f"{gv('sobel_track_causal', 'doctoral', 'success_pct')}% for gradient magnitude. The GNSS-aided replay of the "
    "doctoral study, which scales every frame with its own GNSS altitude and GNSS-position DEM value, reaches "
    f"{gv('m4e300_track_aided', 'doctoral', 'success_pct')}% and {gv('f48k4s1_track_aided', 'doctoral', 'success_pct')}%; "
    f"the difference to the causal replay ({X('aided_minus_causal_m4', '?')} and {X('aided_minus_causal_f48', '?')} "
    "points) is the cost of replacing current-frame GNSS by barometric height and the predicted position, mostly over "
    "sloping rural terrain. No GNSS enters the causal loop after initialization. Table 4 lists the causal replay route "
    "by route for every representation, and Fig. 6 shows two routes. On the five Mavic 2 Pro urban "
    "routes the Model 4 lineage is correct for "
    f"{rng('m4e300_track_causal', PRO_URB, 'success_pct')}% of the eligible frames and checkpoint 1850 for "
    f"{rng('f48k4s1_track_causal', PRO_URB, 'success_pct')}%, with median errors of " +
    (f"{rng('m4e300_track_causal', PRO_URB, 'median_m', 0)} m for both."
     if rng('m4e300_track_causal', PRO_URB, 'median_m', 0) == rng('f48k4s1_track_causal', PRO_URB, 'median_m', 0) else
     f"{rng('m4e300_track_causal', PRO_URB, 'median_m', 0)} m and {rng('f48k4s1_track_causal', PRO_URB, 'median_m', 0)} m.")))
if X('preproc_par'):
    insert_par_after(p, X('preproc_par'))
i, p = find_par(d, 'Table 4. Route-level results for the eight urban flights')
set_par(p, "Table 4. Route-level results of the causal (GNSS-denied after initialization) replay, 30 m threshold. Captured: frames in the flight; Eligible: eligible frames (Section 3.7), identical for all replays. Success rate = correct / eligible for grayscale NCC (Gray), gradient magnitude (Grad.), the Model 4 lineage (M4) and f48-k4-s1 checkpoints 1850 and 1550; Prec. = correct / accepted and Med. = median error of accepted fixes for the Model 4 lineage. Camera: Mavic 2 Zoom (FC2204) or Mavic 2 Pro (L1D-20c).")
hdr, rows = RT.route_table(R)
t = build_table(d, d.tables[3], hdr, rows, [900, 1080, 760, 900, 800, 600, 600, 600, 600, 600, 700, 660],
                group_header=[('', 5), ('Success rate (%)', 5), ('Model 4 lineage', 2)])
replace_table(d, d.tables[3], t)
i, p = find_par(d, 'a Date inferred from the flight log')
set_par(p, ("^{h} Additional flight, not reported in the doctoral study. Rural 4 has no eligible frame (all frames smaller "
            "than a 544 px crop at map resolution); 106 frames of Rural 5 lie outside the DEM, and in tracking almost all of its "
            "eligible frames are lost because the predicted footprint leaves the DEM. Dates and "
            "cameras are read from the image metadata. Per-route RMSE, MAE, availability and counts are given in the "
            "released result tables."))
i, p = find_par(d, 'Table 5. Route-level results for the three rural flights')
set_par(p, "Table 5. Sensitivity of the causal replay of the Model 4 lineage to heading, height and initialization errors (eleven doctoral routes, 30 m threshold; change relative to the nominal replay). Heading offsets are added to the recorded yaw; height offsets scale the height above ground (and hence the frame scale); the initial position is displaced north-east.")
hdr, rows = RT.sensitivity_table(G)
t = build_table(d, d.tables[4], hdr, rows, [2800, 1000, 1200, 1000, 1000, 1000])
replace_table(d, d.tables[4], t)
i, p = find_par(d, 'a The archived confusion counts of Rural 2 sum to 231 frames')
set_par(p, X('sens_note', "Each row is a complete replay of the eleven routes with the perturbation applied to every frame (heading, height) or to the initialization only."))
i, p = find_par(d, 'Rural performance was lower')
set_par(p, X('failure_par', 'FAILURE_ANALYSIS_PLACEHOLDER'))
i, p = find_par(d, 'Fig. 6. Flight routes over Ürgüp.')
set_par(p, X('fig6_caption', "Fig. 6. Causal replay (GNSS read only at the first frame) of two routes with the benchmark-selected f48-k4-s1 model. Green: GPS track; yellow: accepted fixes within 30 m; red: accepted fixes beyond 30 m. (a) Urban 5 (Mavic 2 Pro). (b) Rural 1. Basemap: Google Maps (a) and Bing Maps (b) satellite imagery, © Google, © Microsoft."))

if X('fig6_png'):
    from PIL import Image
    part = None
    idx, cap = find_par(d, 'Fig. 6.')
    png = X('fig6_png')
    replace_picture_before(d, 'Fig. 6.', png)

# ------------------------------------------------------------------ 4.3 / 4.4 references to the flown model
i, p = find_par(d, 'Fig. 7 and Table 6 summarize the main benchmark result.')
p_txt = p.text.replace('The thesis-lineage stride-2 configuration (f48-k5-s2, archived checkpoint 500; Section 4.2)',
                       'The thesis-lineage stride-2 configuration (f48-k5-s2, archived checkpoint 500)')
set_par(p, p_txt)

# ------------------------------------------------------------------ 4.3 gradient magnitude on the district benchmark
import copy
if X('bench_sobel_par'):
    i, p = find_par(d, 'The wider district mosaics (Fig. 7(b)')
    insert_par_after(p, X('bench_sobel_par'))
    i, p = find_par(d, 'The benchmark also exposes an asymmetry')
    set_par(p, X('asym_par'))
    i, p = find_par(d, 'Fig. 7. Success rate at 25 m')
    set_par(p, p.text.rstrip() + X('fig7_add'))
    i, p = find_par(d, 'Table 7 lists the best checkpoint')
    set_par(p, p.text.rstrip() + ' ' + X('t7_sentence'))
    i, p = find_par(d, 'Table 7. Best checkpoint per training lineage')
    set_par(p, X('t7_caption'))
    i, p = find_par(d, 'Controlled comparisons')
    set_par(p, X('contrib_baselines'))
    # Table 7 (d.tables[6]): append the gradient-magnitude row below grayscale, moving the closing rule down
    t7 = d.tables[6]
    old_last = t7.rows[-1]._tr
    new_tr = copy.deepcopy(old_last)
    for tc in old_last.findall(qn('w:tc')):
        bot = tc.find(qn('w:tcPr')).find(qn('w:tcBorders')).find(qn('w:bottom'))
        if bot is not None:
            bot.set(qn('w:val'), 'none'); bot.set(qn('w:sz'), '0'); bot.set(qn('w:color'), 'FFFFFF')
    old_last.addnext(new_tr)
    row = t7.rows[-1]
    assert len(row.cells) == len(X('t7_row')), 'Table 7 column count changed'
    for c, txt in zip(row.cells, X('t7_row')):
        par = c.paragraphs[0]
        par.runs[0].text = txt
        for r_ in par.runs[1:]: r_._r.getparent().remove(r_._r)
        for extra in c.paragraphs[1:]: extra._p.getparent().remove(extra._p)
if X('texture_par'):
    i, p = find_par(d, 'Fig. 9 traces success as a function of training checkpoint')
    insert_par_after(p, X('texture_par'))
if X('recon_par'):
    i, p = find_par(d, 'The probe also revealed a practical sensitivity')
    insert_par_after(p, X('recon_par'))

# ------------------------------------------------------------------ abstract, discussion, conclusions, data
for key, prefix in (('abstract', 'Unmanned aerial vehicles (UAVs) rely on global navigation satellite systems'),
                    ('disc_interp', 'Interpretation. Both evaluation protocols'),
                    ('disc_stride', 'Stride-1 versus stride-2.'),
                    ('disc_lit', 'Comparison with the literature.'),
                    ('disc_valid', 'Validity boundaries.'),
                    ('disc_future', 'Future work.'),
                    ('conclusions', 'We have described a compact cross-autoencoder'),
                    ('data_avail', 'The code supporting this article is openly available'),
                    ('front_contrib_flight', None)):
    if X(key) and prefix:
        i, p = find_par(d, prefix)
        set_par(p, X(key))

d.save(OUT)
print('saved', OUT)
