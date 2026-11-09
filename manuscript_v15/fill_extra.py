#!/usr/bin/env python3
"""Numbers for the results-dependent paragraphs of v15.

Usage: python fill_extra.py OUT_numbers.json RUNS_SPEC [RUNS_SPEC ...] [--texture SUMMARY.csv] [--recon RECON.json]
                                                                       [--eq1 RUNS_SPEC ...]
RUNS_SPEC as in make_tables.py (folder[@prefix,prefix]). Frames, eligibility and outcome definitions are those of
make_tables.py (common eligible set from gray_oracle). The camera offset is estimated per camera from the accepted
fixes (error <= 100 m) of the other doctoral routes flown with the same camera (leave-one-route-out) in the body
frame of the vehicle (forward / right of the recorded yaw).
"""
import os, sys, json, math, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as MT

THR = 30.0
R_E = 6378137.0


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and (math.isnan(x))): return '?'
    return f'{x:,.{nd}f}'


def grp(df, run, group='doctoral'):
    g = df[(df.run == run) & (df.group == group)]
    return MT.summarize(g, THR) if len(g) else None


def body_offsets(a):
    """a: accepted fixes with lat/lon true/est and yaw -> adds dE, dN, fwd, right (metres)."""
    lat0 = np.radians(a['lat_true'].values)
    dN = np.radians(a['lat_est'].values - a['lat_true'].values) * R_E
    dE = np.radians(a['lon_est'].values - a['lon_true'].values) * R_E * np.cos(lat0)
    y = np.radians(a['yaw'].values)
    a = a.assign(dE=dE, dN=dN, fwd=dE * np.sin(y) + dN * np.cos(y), right=dE * np.cos(y) - dN * np.sin(y))
    return a


def bias_loo(df, run):
    d = df[df.run == run]
    a = body_offsets(d[d['accepted'] & d['lat_est'].notna()].copy())
    a['doctoral'] = ~a.route.str.endswith('heldout')
    out_rows = []
    corr = []
    for r, g in a.groupby('route'):
        cal = a[(a.camera == g.camera.iloc[0]) & (a.route != r) & a.doctoral & (a.err_m <= 100)]
        bf, br = cal.fwd.median(), cal.right.median()
        y = np.radians(g['yaw'].values)
        cE = bf * np.sin(y) + br * np.cos(y); cN = bf * np.cos(y) - br * np.sin(y)
        e_corr = np.hypot(g.dE.values - cE, g.dN.values - cN)
        corr.append(pd.DataFrame({'route': r, 'file': g.file.values, 'err_corr': e_corr}))
        out_rows.append(dict(route=r, camera=g.camera.iloc[0], accepted=len(g), bias_fwd=bf, bias_right=br,
                             prec_before=100 * (g.err_m <= THR).mean(), prec_after=100 * (e_corr <= THR).mean(),
                             median_before=g.err_m.median(), median_after=float(np.median(e_corr))))
    per_route = pd.DataFrame(out_rows)
    c = pd.concat(corr, ignore_index=True).merge(a[['route', 'file', 'err_m', 'doctoral']], on=['route', 'file'])
    el = d[d['eligible_m']]
    res = {}
    for name, sel in (('doctoral', c.doctoral), ('additional', ~c.doctoral)):
        s = c[sel]
        n_el = int(el[(~el.route.str.endswith('heldout')) if name == 'doctoral' else el.route.str.endswith('heldout')].shape[0])
        res[name] = dict(accepted=len(s), prec_before=100 * (s.err_m <= THR).mean() if len(s) else np.nan,
                         prec_after=100 * (s.err_corr <= THR).mean() if len(s) else np.nan,
                         succ_before=100 * (s.err_m <= THR).sum() / n_el if n_el else np.nan,
                         succ_after=100 * (s.err_corr <= THR).sum() / n_el if n_el else np.nan,
                         med_before=float(s.err_m.median()) if len(s) else np.nan,
                         med_after=float(s.err_corr.median()) if len(s) else np.nan)
    cal = a[a.doctoral & (a.err_m <= 100)]
    cam = cal.groupby('camera')[['fwd', 'right']].median()
    return per_route, res, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('runs', nargs='+')
    ap.add_argument('--texture', default=''); ap.add_argument('--recon', default='')
    ap.add_argument('--eq1', nargs='*', default=[])
    ap.add_argument('--bias-run', default='')
    ap.add_argument('--sobel-bench', default='', help='sobel_summary.json of benchmark/deney_araclari/sobel_benchmark.py')
    a = ap.parse_args()
    df = MT.annotate(MT.load(a.runs), THR)
    N = {}
    runs = sorted(df.run.unique())
    short = {'gray': 'gray', 'sobel': 'sobel', 'm4e300': 'm4', 'f48k4s1': 'f48', 'f48k4s1c1550': 'c1550'}
    modes = {'oracle': 'or', 'track_aided': 'ag', 'track_causal': 'ca'}
    for run in runs:
        parts = run.split('_', 1)
        if parts[0] not in short or parts[1] not in modes: continue
        for group, gs in (('doctoral', ''), ('heldout', '_ho')):
            s = grp(df, run, group)
            if s is None: continue
            k = f'{modes[parts[1]]}_{short[parts[0]]}{gs}'
            N[k] = fmt(s.success_pct); N[k + '_prec'] = fmt(s.precision_pct); N[k + '_avail'] = fmt(s.availability_pct)
            N[k + '_med'] = fmt(s.median_m); N[k + '_unres'] = fmt(100 - s.availability_pct); N[k + '_elig'] = f'{int(s.eligible):,}'
            N['_raw_' + k] = float(s.success_pct)
            N[k + '_x'] = f'{s.success_pct:.4f}'; N[k + '_prec_x'] = f'{s.precision_pct:.4f}'
    for m in ('m4', 'f48', 'gray'):
        if f'_raw_ag_{m}' in N and f'_raw_ca_{m}' in N:
            dlt = N[f'_raw_ag_{m}'] - N[f'_raw_ca_{m}']
            N[f'aided_minus_causal_{m}'] = f'{dlt:.1f}'
    # Rural 2 eligibility
    r2 = df[(df.run == 'gray_oracle') & (df.route == 'Rural2')]
    N['r2_inel'] = str(int(len(r2) - r2['eligible_m'].sum()))
    N['r2_cap'] = str(len(r2))
    for suf in ('yaw+5', 'yaw-5', 'yaw+10', 'yaw-10', 'alt+5pct', 'alt-5pct', 'alt+10pct', 'alt-10pct', 'init+200m', 'init+500m'):
        s_ = grp(df, f'm4e300_track_causal_{suf}')
        if s_ is not None: N[f'sens_{suf}'] = fmt(s_.success_pct)
    u5 = df[(df.run == 'm4e300_track_causal') & (df.route == 'Urban5') & df['eligible_m']]
    if len(u5): N['fig6_u5'] = f"{100 * u5['correct'].mean():.1f}%"
    # camera offset (leave-one-route-out) for the CAE causal runs and grayscale
    for run, key in (('f48k4s1_track_causal', 'f48'), ('m4e300_track_causal', 'm4'), ('gray_track_causal', 'gray'),
                     ('f48k4s1c1550_track_causal', 'c1550'), ('sobel_track_causal', 'sobel')):
        if run not in runs: continue
        per_route, res, cam = bias_loo(df, run)
        per_route.round(1).to_csv(os.path.splitext(a.out)[0] + f'_bias_{key}.csv', index=False)
        N[f'p_before_{key}'] = fmt(res['doctoral']['prec_before']); N[f'p_after_{key}'] = fmt(res['doctoral']['prec_after'])
        N[f'pa_before_{key}'] = fmt(res['additional']['prec_before']); N[f'pa_after_{key}'] = fmt(res['additional']['prec_after'])
        N[f's_before_{key}'] = fmt(res['doctoral']['succ_before']); N[f's_after_{key}'] = fmt(res['doctoral']['succ_after'])
        N[f'med_before_{key}'] = fmt(res['doctoral']['med_before']); N[f'med_after_{key}'] = fmt(res['doctoral']['med_after'])
        for camname, cshort in (('L1D-20c', 'l1d'), ('FC2204', 'fc')):
            if camname in cam.index:
                N[f'b_{cshort}_f_{key}'] = fmt(cam.loc[camname, 'fwd']); N[f'b_{cshort}_r_{key}'] = fmt(cam.loc[camname, 'right'])
    # texture probe
    if a.texture and os.path.exists(a.texture):
        t = pd.read_csv(a.texture)
        for st in ('low', 'regular'):
            for rep in ('gray', 'sobel', 'cae'):
                s = t[(t.stratum == st) & (t.representation == rep)]
                N[f'{"low" if st == "low" else "reg"}_{rep}'] = fmt(s.success_pct.mean()) if len(s) else '?'
        low = t[(t.stratum == 'low') & (t.representation == 'gray')]
        N['n_low_g'] = str(int(low[low.direction == 'google_to_bing'].n.sum())) if len(low) else '?'
        N['n_low_b'] = str(int(low[low.direction == 'bing_to_google'].n.sum())) if len(low) else '?'
        reg = t[(t.stratum == 'regular') & (t.representation == 'gray')]
        N['n_reg'] = str(int(reg.n.max())) if len(reg) else '?'
    if a.recon and os.path.exists(a.recon):
        N.update({f'recon_{k}': fmt(v) for k, v in json.load(open(a.recon)).items()})
    # gradient magnitude on the district benchmark (global search, identical queries to Table 6)
    if a.sobel_bench and os.path.exists(a.sobel_bench):
        S = json.load(open(a.sobel_bench))
        M = {'raw': 'gray', 'cae': 'cae', 'sobel': 'sob'}
        V = {'clean': 'c', 'hard_v1': 'h'}
        for v, vs in V.items():
            for m, ms in M.items():
                N[f'sb_{ms}_{vs}'] = fmt(S['mean'][v][m]); N[f'sb_{ms}_{vs}_x'] = f"{S['mean'][v][m]:.4f}"
                N[f'sb_{ms}_{vs}_0'] = f"{S['mean'][v][m]:.0f}"
                for d in ('gb', 'bg'):
                    N[f'sb_{ms}_{d}_{vs}'] = fmt(S['by_direction'][f'{d}|{v}'][m]); N[f'sb_{ms}_{d}_{vs}_0'] = f"{S['by_direction'][f'{d}|{v}'][m]:.0f}"
                for half in ('north', 'south'):
                    N[f'sb_{half[0]}_{ms}_{vs}'] = fmt(S['halves'][half][f'{v}|{m}'])
                for tert, ts in (('low', 'lo'), ('high', 'hi')):
                    N[f'sb_{ts}_{ms}_{vs}'] = fmt(S['entropy_tertiles'][f'{tert}|{v}'][m])
            P = S['paired'][v]
            for a_, b_, ks in (('sobel', 'cae', 'cae'), ('sobel', 'raw', 'gray'), ('cae', 'raw', 'cg')):
                d_ = P[f'{a_}_minus_{b_}']
                N[f'sb_d_{ks}_{vs}'] = f"{d_['pp']:+.1f}"; N[f'sb_d_{ks}_{vs}_x'] = f"{d_['pp']:.4f}"
                N[f'sb_ci_{ks}_{vs}'] = f"{d_['ci95'][0]:.1f} to {d_['ci95'][1]:.1f}"
                N[f'sb_cilo_{ks}_{vs}'] = fmt(d_['ci95'][0]); N[f'sb_cihi_{ks}_{vs}'] = fmt(d_['ci95'][1])
                N[f'sb_cilo_{ks}_{vs}_x'] = f"{d_['ci95'][0]:.4f}"; N[f'sb_cihi_{ks}_{vs}_x'] = f"{d_['ci95'][1]:.4f}"
            N[f'sb_only_{vs}'] = f"{P['sobel_only']:,}"; N[f'sb_caeonly_{vs}'] = f"{P['cae_only']:,}"
        for m, ms in M.items():
            N[f'sb_drop_{ms}'] = fmt(S['mean']['clean'][m] - S['mean']['hard_v1'][m])
            N[f'sb_med_{ms}'] = fmt(S['median_error_of_successes_m'][m])
        N['sb_rej'] = str(S['sobel_rejected']); N['sb_n'] = f"{S['n_per_direction_variant']:,}"
        N['sb_searches'] = f"{4 * S['n_per_direction_variant']:,}"
        N['sb_raw_diff'] = str(S['raw_differs_from_table6'])
        N['sb_blocks_n'] = str(S['halves']['north']['blocks']); N['sb_blocks_s'] = str(S['halves']['south']['blocks'])
    # equalisation sensitivity (eq = 1)
    if a.eq1:
        d1 = MT.annotate(MT.load(a.eq1), THR)
        for run in sorted(d1.run.unique()):
            parts = run.split('_', 1)
            if parts[0] not in short or parts[1] not in modes: continue
            s = grp(d1, run)
            if s is not None: N[f'eq1_{modes[parts[1]]}_{short[parts[0]]}'] = fmt(s.success_pct)
            # difference eq1 - eq0 on the doctoral routes present in both
            r1 = set(d1[(d1.run == run) & (d1.group == 'doctoral')].route); r0 = set(df[(df.run == run) & (df.group == 'doctoral')].route)
            common = r1 & r0
            if common:
                s1 = MT.summarize(d1[(d1.run == run) & d1.route.isin(common)], THR)
                s0 = MT.summarize(df[(df.run == run) & df.route.isin(common)], THR)
                N[f'eq1_delta_{modes[parts[1]]}_{short[parts[0]]}'] = f'{s1.success_pct - s0.success_pct:.1f}'
                N[f'eq1_routes_{modes[parts[1]]}_{short[parts[0]]}'] = str(len(common))
    json.dump({k: v for k, v in N.items() if not k.startswith('_raw_')}, open(a.out, 'w'), indent=1, ensure_ascii=False)
    for k in sorted(N):
        if not k.startswith('_raw_'): print(f'{k:28s} {N[k]}')


if __name__ == '__main__':
    main()
