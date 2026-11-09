#!/usr/bin/env python3
"""Build every flight table of the article from the per-frame replay records (flight_replay/runs/*.csv).

Definitions (threshold THR = 30 m):
  captured     frames in the route folder (all written to the CSV)
  eligible     frames whose footprint lies inside the reference map and is at least one crop at map resolution,
               judged at the true position (GNSS position and altitude) in the reference run REF_RUN (gray_oracle).
               The same frame set is used for every run, so all runs share one denominator; a frame that a
               tracking run could not attempt (e.g. predicted footprint off the map) counts as unresolved
  accepted     eligible frames with an accepted position (resolved = 1)
  correct      accepted and error <= THR
  wrong        accepted and error  > THR
  unresolved   eligible - accepted (no consensus, or rejected by the motion gate)
  success      correct / eligible      (headline reliability figure)
  availability accepted / eligible
  precision    correct / accepted
  median/MAE/RMSE/P90 over accepted fixes
Usage: python make_tables.py OUT_DIR RUNS_DIR[@prefix,prefix] [RUNS_DIR[@prefix] ...] [--thr 30] [--ref gray_oracle]
       e.g. make_tables.py tables flight_replay/runs_ne flight_replay/runs@gray_
"""
import os, sys, glob, argparse, math
import numpy as np, pandas as pd

ELIG = {'', 'no_consensus', 'gate'}


def load(specs):
    frames = []
    for spec in specs:
        folder, _, pref = spec.partition('@')
        prefixes = [x for x in pref.split(',') if x]
        for f in sorted(glob.glob(os.path.join(folder, '*.csv'))):
            name = os.path.splitext(os.path.basename(f))[0]
            if prefixes and not any(name.startswith(x) for x in prefixes): continue
            d = pd.read_csv(f, low_memory=False)
            if 'run' not in d.columns or len(d) == 0: continue
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(['run', 'route', 'file'], keep='first')
    df['reason'] = df['reason'].fillna('')
    return df


def annotate(df, thr, ref_run='gray_oracle'):
    attempted = df['reason'].isin(ELIG) & (df['eligible'] == 1)
    ref = df[(df['run'] == ref_run) & attempted]
    if len(ref):
        keys = set(zip(ref['route'], ref['file'].str.lower()))
        df['eligible_m'] = [k in keys for k in zip(df['route'], df['file'].str.lower())]
    else:
        print('warning: reference run', ref_run, 'not found; per-run eligibility used')
        df['eligible_m'] = attempted
    df['not_attempted'] = df['eligible_m'] & ~attempted
    df['accepted'] = (df['resolved'] == 1) & df['eligible_m'] & attempted
    df['correct'] = df['accepted'] & (df['err_m'] <= thr)
    df['wrong'] = df['accepted'] & (df['err_m'] > thr)
    df['group'] = np.where(df['route'].str.endswith('heldout'), 'heldout', 'doctoral')
    df['area'] = np.where(df['route'].str.startswith('Urban'), 'urban', 'rural')
    return df


def summarize(g, thr=30.0):
    acc = g[g['accepted']]
    e = acc['err_m'].values.astype(float)
    n_el = int(g['eligible_m'].sum()); n_acc = int(g['accepted'].sum()); n_ok = int(g['correct'].sum())
    n_wrong = int(g['wrong'].sum())
    out = dict(captured=len(g), eligible=n_el, accepted=n_acc, correct=n_ok, wrong=n_wrong,
               unresolved=n_el - n_acc, not_attempted=int(g['not_attempted'].sum()),
               gate_rejected=int((g['eligible_m'] & (g['reason'] == 'gate')).sum()),
               success_pct=100 * n_ok / n_el if n_el else np.nan,
               availability_pct=100 * n_acc / n_el if n_el else np.nan,
               precision_pct=100 * n_ok / n_acc if n_acc else np.nan,
               median_m=float(np.median(e)) if len(e) else np.nan,
               mae_m=float(np.mean(e)) if len(e) else np.nan,
               rmse_m=float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
               p90_m=float(np.percentile(e, 90)) if len(e) else np.nan,
               max_m=float(np.max(e)) if len(e) else np.nan,
               sec_per_frame=float(g['sec'].mean()) if 'sec' in g and g['sec'].notna().any() else np.nan)
    # internal consistency check (Cauchy-Schwarz): RMSE >= sqrt(S_wrong^2 / (n_wrong * n_acc))
    if n_wrong:
        s_wrong = float(e[e > thr].sum())
        out['rmse_lower_bound_m'] = math.sqrt(s_wrong ** 2 / (n_wrong * n_acc))
        out['consistent'] = bool(out['rmse_m'] + 1e-9 >= out['rmse_lower_bound_m'])
    else:
        out['rmse_lower_bound_m'] = np.nan; out['consistent'] = True
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('runs', nargs='+')
    ap.add_argument('--thr', type=float, default=30.0); ap.add_argument('--ref', default='gray_oracle')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = annotate(load(a.runs), a.thr, a.ref)
    keys = ['run', 'model', 'mode']
    S = lambda g: summarize(g, a.thr)
    by_route = df.groupby(keys + ['route'], group_keys=False).apply(S).reset_index()
    by_group = df.groupby(keys + ['group'], group_keys=False).apply(S).reset_index()
    by_group_area = df.groupby(keys + ['group', 'area'], group_keys=False).apply(S).reset_index()
    for name, t in (('by_route', by_route), ('by_group', by_group), ('by_group_area', by_group_area)):
        t.round(6).to_csv(os.path.join(a.out, f'{name}.csv'), index=False)
    # frame manifest: one row per captured frame with its outcome in every run
    base = df.sort_values(['route', 'idx']).drop_duplicates(['route', 'file'])[
        ['route', 'idx', 'file', 'datetime', 'camera', 'lat_true', 'lon_true', 'gps_alt', 'rel_alt', 'yaw']]
    piv = df.assign(outcome=np.where(~df['eligible_m'], 'not_eligible:' + df['reason'],
                                     np.where(df['correct'], 'correct', np.where(df['wrong'], 'wrong',
                                              np.where(df['not_attempted'], 'not_attempted:' + df['reason'], 'unresolved:' + df['reason'])))))
    piv_o = piv.pivot_table(index=['route', 'file'], columns='run', values='outcome', aggfunc='first')
    piv_e = piv.pivot_table(index=['route', 'file'], columns='run', values='err_m', aggfunc='first').add_prefix('err_m:')
    man = base.merge(piv_o.reset_index(), on=['route', 'file'], how='left').merge(piv_e.reset_index(), on=['route', 'file'], how='left')
    man.to_csv(os.path.join(a.out, 'frame_manifest_with_outcomes.csv'), index=False)
    reasons = df.groupby(['run', 'route', 'reason']).size().unstack(2, fill_value=0)
    reasons.to_csv(os.path.join(a.out, 'frame_reasons.csv'))
    bad = by_route[~by_route['consistent'].astype(bool)]
    print('routes x runs:', len(by_route), '| internal-consistency violations:', len(bad))
    with open(os.path.join(a.out, 'summary.md'), 'w') as f:
        f.write(f'# Replay summary (threshold {a.thr:.0f} m)\n\n## By group\n\n' + by_group.round(1).to_markdown(index=False) + '\n\n')
        f.write('## By group and area\n\n' + by_group_area.round(1).to_markdown(index=False) + '\n\n')
        f.write('## By route\n\n' + by_route.round(1).to_markdown(index=False) + '\n')
    print(by_group.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
