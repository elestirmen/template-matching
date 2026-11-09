#!/usr/bin/env python3
"""Redraw the data figures of the manuscript (IJEG: 18 cm wide, >= 400 dpi) from the benchmark outputs.

Figure 1  pipeline diagram (drawn here; text follows Sections 2.1-2.6 of the IJEG version)
Figure 7  success at 25 m versus search radius: (a) core mosaics, 300 queries/direction; (b) district, 4,000 queries
Figure 8  (a) screening of 545 checkpoints (300 queries, global); (b) peak-separation probe (margin_gate)
Figure 9  global success versus training checkpoint for three lineages (2,000 queries/direction)

Usage (from manuscript_v15): python ijeg/make_figs.py [OUT_DIR]      default OUT_DIR = ijeg/figs
The original PNGs of September 2026 were produced in a Claude session whose script was not kept; this file
reproduces them from the archived summaries so that they can be regenerated. Keyboard hyphens only in labels."""
import os, sys, json, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'benchmark', 'outputs'))
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figs')
os.makedirs(OUT_DIR, exist_ok=True)
W_IN = 18.0 / 2.54          # 18 cm
DPI = 450
BLUE, ORANGE, GREY, GREEN, VERM = '#0072B2', '#E69F00', '#7f7f7f', '#009E73', '#D55E00'
MODES = ['roi_500m', 'roi_1000m', 'roi_2000m', 'roi_4000m', 'roi_8000m', 'global']
XT = ['0.5', '1', '2', '4', '8', 'global']
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9.5, 'legend.fontsize': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linewidth': 0.5, 'savefig.facecolor': 'white'})


def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=DPI, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    print(f'{name}: {w} x {h} px, {w / (18 / 2.54):.0f} dpi at 18 cm')


def mean_by_mode(df, model_id, variant):
    d = df[(df.model_id == model_id) & (df.query_variant == variant) & (df.search_mode.isin(MODES))].copy()
    for c in ('success_25m_ci95_low', 'success_25m_ci95_high'):   # the core run (Aug 2026) has no bootstrap columns
        if c not in d: d[c] = np.nan
    g = d.groupby('search_mode').agg(s=('success_25m', 'mean'), lo=('success_25m_ci95_low', 'mean'),
                                     hi=('success_25m_ci95_high', 'mean')) * 100
    return g.reindex(MODES)


# ------------------------------------------------------------------------------------------ Figure 7
def fig7():
    core = pd.read_csv(os.path.join(OUTS, 'run_20260809_225158', 'summary.csv'))
    dist = pd.read_csv(os.path.join(OUTS, 'urgup_cevresi_models_4000', 'summary.csv'))
    cae_d = [m for m in dist.model_id.unique() if m.endswith('cb62bfb86ad0')][0]          # checkpoint 1850 (Table 6)
    core_models = [('tpu_model_f48_k4_epoch_01100_sigmoid__1__1', 'CAE f48-k4-s1', BLUE),
                   ('GPU_model_f48_k5_epoch_00500_sigmoid__2__2', 'CAE f48-k5-s2', ORANGE),
                   ('RAW_BASELINE', 'Grayscale NCC', GREY)]
    for m, _, _ in core_models: assert m in set(core.model_id), m
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, W_IN * 1009 / 2128), sharey=True)
    x = np.arange(len(MODES))
    for ax, title, df, models, band in ((axes[0], '(a) Ürgüp core, 300 queries/direction', core, core_models, False),
                                        (axes[1], '(b) Wider Ürgüp region, 4000 queries/direction', dist,
                                         [(cae_d, 'CAE f48-k4-s1', BLUE), ('RAW_BASELINE', 'Grayscale NCC', GREY)], True)):
        for mid, lab, col in models:
            for var, ls in (('clean', '-'), ('hard_v1', '--')):
                g = mean_by_mode(df, mid, var)
                ax.plot(x, g.s.values, ls, color=col, marker='o', ms=4, lw=1.8, label=f'{lab} ({var})')
                if band: ax.fill_between(x, g.lo.values, g.hi.values, color=col, alpha=0.12, lw=0)
        ax.set_title(title, loc='left'); ax.set_xticks(x); ax.set_xticklabels(XT); ax.set_xlabel('Search radius (km)')
        ax.set_ylim(30, 100)
    axes[0].set_ylabel('Success rate @ 25 m (%)')
    h, l = axes[0].get_legend_handles_labels()
    order = [0, 1, 2, 3, 4, 5]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(); save(fig, 'fig7_search_radius.png')


# ------------------------------------------------------------------------------------------ Figure 8
def fig8():
    scr = pd.read_csv(os.path.join(OUTS, 'urgup_bes_nokta_global_100', 'summary.csv'))
    scr = scr[scr.search_mode == 'global']
    raw = scr[scr.model_id == 'RAW_BASELINE'].success_25m.mean() * 100
    per = scr[scr.model_id != 'RAW_BASELINE'].groupby('model_id').success_25m.mean() * 100
    gate = json.load(open(os.path.join(OUTS, 'margin_gate', 'validate.json')))['results']
    def pick(prefix):
        c = {k: v for k, v in gate.items() if k.startswith(prefix)}
        return max(c.values(), key=lambda r: r['separation_z'])
    bars = [('Grayscale\nNCC', gate['RAW']), ('f32-k5-s1\n(22k par.)', pick('01_deneme_6')),
            ('f32-k5-s1\n(23k par.)', pick('02_modeller___15_10_2022')), ('f48-k4-s1\n(50k par.)', pick('_03_model_tpu_f48_k4'))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_IN, W_IN * 841 / 2136), gridspec_kw={'width_ratios': [1.05, 1]})
    a.hist(per.values, bins=np.arange(0, 101, 5), color=BLUE, edgecolor='white', lw=0.5)
    a.axvline(raw, color=VERM, lw=2)
    # label on a white ground, right-aligned just left of the bin that holds the baseline so that it never
    # covers a bar top (the 70-75% bin is the tallest one)
    a.text(5 * np.floor(raw / 5) - 1.0, a.get_ylim()[1] * 0.97, f'grayscale NCC\n{raw:.1f}%', color=VERM, ha='right',
           va='top', fontsize=8.5, bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=5)
    a.set_title(f'(a) Architecture / checkpoint screening, 300 queries', loc='left')
    a.set_xlabel('Mean global success @ 25 m, clean & hard_v1 (%)'); a.set_ylabel(f'Checkpoints (n = {len(per)})')
    a.set_xlim(-2, 102)
    xs = np.arange(len(bars)); w = 0.38
    t = [r['truth_ncc_mean'] for _, r in bars]; f = [r['wrong_ncc_mean'] for _, r in bars]
    b.bar(xs - w / 2, t, w, color=GREEN, label='NCC at true location'); b.bar(xs + w / 2, f, w, color=VERM, label='strongest false peak')
    for i, (_, r) in enumerate(bars):
        b.text(i, max(t[i], f[i]) + 0.012, f"z = {r['separation_z']:.1f}", ha='center', fontsize=8.5)
    b.set_xticks(xs); b.set_xticklabels([n for n, _ in bars], fontsize=8.5); b.set_ylim(0, 0.40)
    b.set_ylabel('Mean NCC score'); b.set_title(f"(b) Peak separation, {gate['RAW']['samples']} probe tiles", loc='left')
    b.legend(frameon=False, loc='upper right'); b.grid(axis='x', visible=False)
    fig.tight_layout(); save(fig, 'fig8_screening_separation.png')


# ------------------------------------------------------------------------------------------ Figure 9
def fig9():
    sw = pd.read_csv(os.path.join(OUTS, 'epoch_sweep_2000', 'summary.csv'))
    sw = sw[sw.search_mode == 'global']
    ep = pd.read_csv(os.path.join(OUTS, 'epoch_sweep_2000', 'all_epoch_results.csv'))[['evaluated_model_id', 'epoch', 'training_lineage']]
    d = sw.merge(ep.drop_duplicates('evaluated_model_id'), left_on='model_id', right_on='evaluated_model_id', how='left')
    raw = sw[sw.model_id == 'RAW_BASELINE'].groupby('query_variant').success_25m.mean() * 100
    lineages = [('model_tpu_f48_k4_s1_elu', '(a) f48-k4-s1 (TPU run)'), ('deneme_6__gpu_model_f32_k5', '(b) f32-k5-s1 (GPU run)'),
                ('kosmus_model_f32_k3', '(c) f32-k3-s1')]
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, W_IN * 894 / 2135), sharey=True)
    for ax, (key, title) in zip(axes, lineages):
        g = d[d.training_lineage.fillna('').str.startswith(key)].groupby(['epoch', 'query_variant']).success_25m.mean().unstack() * 100
        assert len(g), key
        ax.plot(g.index, g['clean'], '-o', color=BLUE, ms=3, lw=1.5, label='clean')
        ax.plot(g.index, g['hard_v1'], '-o', color=ORANGE, ms=3, lw=1.5, label='hard_v1')
        ax.axhline(raw['clean'], color=GREY, lw=1.6, label='grayscale NCC (clean)')
        ax.axhline(raw['hard_v1'], color=GREY, lw=1.6, ls='--', label='grayscale NCC (hard_v1)')
        ax.set_title(title, loc='left'); ax.set_xlabel('Checkpoint (epoch index)'); ax.set_ylim(0, 100)
    axes[0].set_ylabel('Global success @ 25 m (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); save(fig, 'fig9_checkpoint_sweep.png')


# ------------------------------------------------------------------------------------------ Figure 1
def fig1():
    fig = plt.figure(figsize=(W_IN, W_IN * 1053 / 1780)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')
    NAVY = '#2E4A6B'; FILL = '#EEF3F9'; ORF, ORE = '#FFF3DC', '#C47A00'; GRF, GRE = '#EAF5EE', '#1E7A3C'
    boxes = []

    def box(x, y, w, h, text, fill=FILL, edge=NAVY, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc=fill, ec=edge, lw=1.6))
        t = ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color='#111111', linespacing=1.3)
        boxes.append((t, x, y, w, h))
    def arrow(p, q, **kw):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, color=NAVY, lw=1.6, shrinkA=0, shrinkB=0, **kw))
    bw, bh, gap, x0, FS = 21.0, 12, 2.0, 2, 7.2
    xs = [x0 + i * (bw + gap) for i in range(4)]
    # offline row
    ax.text(x0, 58, 'Offline (once per mission area)', fontsize=10, fontweight='bold', color=NAVY, va='center')
    y1 = 43
    texts1 = ['Georeferenced satellite\nmosaic (0.298 m/px,\nEPSG:32636)', 'Tiling: 544 × 544 px,\n32 px overlap',
              'Structural transform\n(cross-autoencoder, same\nweights for map and UAV,\nor gradient magnitude)', 'Crop 16 px per edge,\nstitch → structural\nreference raster']
    for i, t in enumerate(texts1):
        box(xs[i], y1, bw, bh, t, *((ORF, ORE) if i == 2 else (FILL, NAVY)), fs=FS)
        if i: arrow((xs[i - 1] + bw, y1 + bh / 2), (xs[i], y1 + bh / 2))
    # online row
    ax.text(x0, 38, 'Online (per UAV frame)', fontsize=10, fontweight='bold', color=NAVY, va='center')
    y2 = 23
    texts2 = ['Nadir RGB frame +\nheading, barometric\naltitude, camera\nintrinsics', 'Undistort, rotate north,\nDEM height → GSD,\nresample to map GSD',
              'Three 544 × 544 crops\n(offsets d ≤ 200 px) →\nsame structural\ntransform', 'Three 512 × 512\ntemplates']
    for i, t in enumerate(texts2):
        box(xs[i], y2, bw, bh, t, *((ORF, ORE) if i == 2 else (FILL, NAVY)), fs=FS)
        if i: arrow((xs[i - 1] + bw, y2 + bh / 2), (xs[i], y2 + bh / 2))
    # bottom row
    y3 = 3
    box(xs[3], y3, bw, bh, 'NCC\n(TM_CCOEFF_NORMED)\nin a window around\nthe predicted position;\nexpanded on failure', GRF, GRE, fs=FS)
    box(xs[1], y3, 2 * bw + gap, bh, 'Three-crop consensus: implied frame centers\nagree within 48 px → mean position → UTM;\nthe window follows the fix through a visual-\nodometry prior; GNSS read only at initialization', GRF, GRE, fs=FS)
    arrow((xs[3], y3 + bh / 2), (xs[2] + bw, y3 + bh / 2))
    # templates down, search space from the reference raster around the right side
    arrow((xs[3] + bw / 2 - 4, y2), (xs[3] + bw / 2 - 4, y3 + bh))
    ax.text(xs[3] + bw / 2 - 3, (y2 + y3 + bh) / 2, 'templates', fontsize=FS, color=NAVY, va='center')
    xr = xs[3] + bw + 1.5
    ax.plot([xs[3] + bw, xr, xr, xs[3] + bw / 2 + 5], [y1 + bh / 2, y1 + bh / 2, y3 + bh + 3, y3 + bh + 3], color=NAVY, lw=1.6, solid_capstyle='round')
    arrow((xs[3] + bw / 2 + 5, y3 + bh + 3), (xs[3] + bw / 2 + 5, y3 + bh))
    ax.text(xr + 0.8, (y1 + y3 + bh) / 2 + 2, 'search space', fontsize=FS, color=NAVY, rotation=90, va='center')
    # every label must sit inside its box with a margin (1.2 units across, 0.8 units vertically)
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    for t, x, y, w, h in boxes:
        (bx0, by0), (bx1, by1) = inv.transform(t.get_window_extent(r).get_points())
        if bx0 < x + 1.2 or bx1 > x + w - 1.2 or by0 < y + 0.8 or by1 > y + h - 0.8:
            raise SystemExit(f'Figure 1: text does not fit its box with margin: {t.get_text()!r} '
                             f'text x {bx0:.1f}-{bx1:.1f} y {by0:.1f}-{by1:.1f}, box x {x:.1f}-{x + w:.1f} y {y:.1f}-{y + h:.1f}')
    save(fig, 'fig1_pipeline.png')


if __name__ == '__main__':
    fig7(); fig8(); fig9(); fig1()
