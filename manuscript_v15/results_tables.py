"""Build the flight-result tables of v15 from make_tables.py outputs (by_group.csv, by_route.csv)."""
import pandas as pd, numpy as np

MODEL_LABEL = {
    'grayscale': 'Grayscale NCC',
    'sobel': 'Gradient magnitude',
    'GPU_model_f48_k5_epoch_00300_sigmoid_(2, 2)_.h5': 'CAE f48-k5-s2 (Model 4 lineage)',
    '03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__cb62bfb86ad0.h5': 'CAE f48-k4-s1 (ckpt 1850)',
    '_03_model_tpu_f48_k4_s1_elu__tpu_model_f48_k4_epoch__epoch__sigmoid__1__7__bdb7daa618c0.h5': 'CAE f48-k4-s1 (ckpt 1550)',
}
RUN_ORDER = [  # (run, mode label)
    ('gray_oracle', 'Oracle'), ('sobel_oracle', 'Oracle'), ('m4e300_oracle', 'Oracle'), ('f48k4s1_oracle', 'Oracle'), ('f48k4s1c1550_oracle', 'Oracle'),
    ('gray_track_aided', 'GNSS-aided'), ('m4e300_track_aided', 'GNSS-aided'), ('f48k4s1_track_aided', 'GNSS-aided'),
    ('gray_track_causal', 'Causal'), ('sobel_track_causal', 'Causal'), ('m4e300_track_causal', 'Causal'), ('f48k4s1_track_causal', 'Causal'), ('f48k4s1c1550_track_causal', 'Causal'),
]
RUN_MODEL = {'gray': 'Grayscale', 'sobel': 'Gradient magnitude', 'm4e300': 'CAE f48-k5-s2 (Model 4)',
             'f48k4s1': 'CAE f48-k4-s1, ckpt 1850', 'f48k4s1c1550': 'CAE f48-k4-s1, ckpt 1550'}
ROUTE_LABEL = {'Urban9_heldout': 'Urban 9', 'Urban10_heldout': 'Urban 10', 'Urban11_heldout': 'Urban 11',
               'Rural4_heldout': 'Rural 4', 'Rural5_heldout': 'Rural 5'}
ROUTE_ORDER = ['Urban1', 'Urban2', 'Urban3', 'Urban4', 'Urban5', 'Urban6', 'Urban7', 'Urban8', 'Rural1', 'Rural2', 'Rural3',
               'Urban9_heldout', 'Urban10_heldout', 'Urban11_heldout', 'Rural4_heldout', 'Rural5_heldout']
ROUTE_META = {  # date (from EXIF), camera
    'Urban1': ('20 Aug 2022', 'Zoom'), 'Urban2': ('22 Aug 2022', 'Zoom'), 'Urban3': ('22 Aug 2022', 'Zoom'),
    'Urban4': ('2 May 2023', 'Pro'), 'Urban5': ('14 Apr 2023', 'Pro'), 'Urban6': ('14 Apr 2023', 'Pro'),
    'Urban7': ('14 Apr 2023', 'Pro'), 'Urban8': ('15 Apr 2023', 'Pro'),
    'Rural1': ('13 Nov 2022', 'Pro'), 'Rural2': ('13 Nov 2022', 'Pro'), 'Rural3': ('13 Nov 2022', 'Pro'),
    'Urban9_heldout': ('2 May 2023', 'Pro'), 'Urban10_heldout': ('20 Aug 2022', 'Zoom'), 'Urban11_heldout': ('20 Aug 2022', 'Zoom'),
    'Rural4_heldout': ('3 Dec 2020', 'Zoom'), 'Rural5_heldout': ('2 Dec 2020', 'Zoom'),
}


def f1(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '-'
    return f'{x:,.{nd}f}'


def f0(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '-'
    return f'{int(round(x)):,}'


def pooled_table(by_group):
    """Rows: mode x representation. Columns: doctoral (eligible, success, availability, precision, median, RMSE)
    and held-out (eligible, success, precision, median)."""
    header = ['Replay', 'Representation', 'Frames', 'Succ. (%)', 'Avail. (%)', 'Prec. (%)', 'Med. (m)', 'RMSE (m)',
              'Frames', 'Succ. (%)', 'Prec. (%)', 'Med. (m)']
    rows = []
    for run, mode in RUN_ORDER:
        g = by_group[by_group['run'] == run]
        if g.empty: continue
        doc = g[g['group'] == 'doctoral']; ho = g[g['group'] == 'heldout']
        D = doc.iloc[0] if len(doc) else None; H = ho.iloc[0] if len(ho) else None
        rows.append([mode, RUN_MODEL[run.split('_')[0]],
                     f0(D.eligible) if D is not None else '-', f1(D.success_pct) if D is not None else '-',
                     f1(D.availability_pct) if D is not None else '-', f1(D.precision_pct) if D is not None else '-',
                     f1(D.median_m) if D is not None else '-', f1(D.rmse_m) if D is not None else '-',
                     f0(H.eligible) if H is not None else '-', f1(H.success_pct) if H is not None else '-',
                     f1(H.precision_pct) if H is not None else '-', f1(H.median_m) if H is not None else '-'])
    return header, rows


def route_table(by_route, runs=('gray_track_causal', 'sobel_track_causal', 'm4e300_track_causal', 'f48k4s1_track_causal',
                                 'f48k4s1c1550_track_causal'), med_run='m4e300_track_causal'):
    header = ['Route', 'Date', 'Camera', 'Captured', 'Eligible', 'Gray', 'Grad.', 'M4', '1850', '1550', 'Prec. (%)', 'Med. (m)']
    rows = []
    for r in ROUTE_ORDER:
        sub = by_route[by_route['route'] == r]
        if sub.empty: continue
        ref = sub.iloc[0]
        name = ROUTE_LABEL.get(r, r.replace('Urban', 'Urban ').replace('Rural', 'Rural '))
        if r.endswith('heldout'): name += '^{h}'
        date, cam = ROUTE_META[r]
        vals = []
        for run in runs:
            x = sub[sub['run'] == run]
            vals.append(f1(x.iloc[0].success_pct) if len(x) and x.iloc[0].eligible > 0 else '-')
        m = sub[sub['run'] == med_run]
        M = m.iloc[0] if len(m) else None
        rows.append([name, date, cam, f0(ref.captured), f0(ref.eligible)] + vals +
                    [f1(M.precision_pct) if M is not None and M.eligible > 0 else '-', f1(M.median_m) if M is not None and M.eligible > 0 else '-'])
    return header, rows


SENS_ORDER = [('', 'None (nominal)'), ('_yaw+5', 'Heading +5°'), ('_yaw-5', 'Heading -5°'), ('_yaw+10', 'Heading +10°'),
              ('_yaw-10', 'Heading -10°'), ('_alt+5pct', 'Height above ground +5%'), ('_alt-5pct', 'Height above ground -5%'),
              ('_alt+10pct', 'Height above ground +10%'), ('_alt-10pct', 'Height above ground -10%'),
              ('_init+200m', 'Initial position 200 m off'), ('_init+500m', 'Initial position 500 m off')]


def sensitivity_table(by_group, model='m4e300'):
    header = ['Perturbation', 'Succ. (%)', 'Change (points)', 'Avail. (%)', 'Prec. (%)', 'Med. (m)']
    rows = []
    base = by_group[(by_group['run'] == f'{model}_track_causal') & (by_group['group'] == 'doctoral')]
    b = float(base.iloc[0].success_pct) if len(base) else float('nan')
    for suf, lab in SENS_ORDER:
        g = by_group[(by_group['run'] == f'{model}_track_causal{suf}') & (by_group['group'] == 'doctoral')]
        if g.empty or (suf and int(g.iloc[0].captured) < int(base.iloc[0].captured)):
            rows.append([lab, '-', '-', '-', '-', '-']); continue
        G = g.iloc[0]
        ch = '' if not suf else f'{G.success_pct - b:+.1f}'
        rows.append([lab, f1(G.success_pct), ch, f1(G.availability_pct), f1(G.precision_pct), f1(G.median_m)])
    return header, rows
