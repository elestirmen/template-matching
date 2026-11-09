#!/usr/bin/env python3
"""Paired uncertainty of the flight success-rate differences between representations.

Frames, eligibility and outcomes are those of make_tables.py (common eligible set from gray_oracle, 30 m).
For a pair of runs A, B in the same replay mode the per-frame difference is d = correct_A - correct_B on the same
eligible frames; the reported difference is the pooled mean of d (percentage points). Consecutive frames are not
independent (overlapping footprints, and in tracking a lost track persists), so the interval is a circular block
bootstrap stratified by route: in every route the eligible frames, in capture order, are resampled in blocks of
L consecutive frames (paired: the same blocks for A and B), and the pooled difference is recomputed; 95% percentile
interval over R replicates. L = 25 is the main setting, L = 10 and 50 and a route-level (cluster) bootstrap are
sensitivity settings. The number of routes on which A beats B is also reported.

Usage: python flight_ci.py OUT_PREFIX RUNS_SPEC [RUNS_SPEC ...] [--reps 10000] [--seed 20261109]
       e.g. flight_ci.py out/flight_ci ../flight_replay/runs_ne ../flight_replay/runs@gray_
Writes OUT_PREFIX.csv and OUT_PREFIX.json.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as MT

THR = 30.0
REPS = {'m4e300': 'm4', 'f48k4s1': 'f48', 'f48k4s1c1550': 'c1550', 'sobel': 'sobel', 'gray': 'gray'}
PAIRS = [('m4e300', 'sobel'), ('f48k4s1', 'sobel'), ('f48k4s1c1550', 'sobel'),
         ('m4e300', 'gray'), ('f48k4s1', 'gray'), ('f48k4s1c1550', 'gray'), ('sobel', 'gray')]
MODES = {'oracle': 'or', 'track_causal': 'ca'}


def paired(dA, dB, L, reps, rng):
    """dA, dB: dict route -> 0/1 arrays in capture order. Circular block bootstrap (blocks of L consecutive frames,
    start uniform on the route, wrapping at the end, so that every frame is equally likely), stratified by route;
    L = 'route' resamples whole routes. Returns point (pp), lo, hi of the pooled difference."""
    routes = sorted(dA)
    diff = {r: dA[r].astype(float) - dB[r].astype(float) for r in routes}
    n_tot = sum(len(diff[r]) for r in routes)
    point = 100 * sum(diff[r].sum() for r in routes) / n_tot
    if L == 'route':
        arr = np.array([diff[r].sum() for r in routes]); ns = np.array([len(diff[r]) for r in routes])
        j = rng.integers(0, len(routes), (reps, len(routes)))
        out = 100 * arr[j].sum(1) / ns[j].sum(1)
    else:
        tot = np.zeros(reps)
        for r in routes:
            x = diff[r]; n = len(x)
            if not n: continue
            k = -(-n // L)
            starts = rng.integers(0, n, (reps, k))
            idx = ((starts[:, :, None] + np.arange(L)[None, None, :]) % n).reshape(reps, -1)[:, :n]
            tot += x[idx].sum(1)
        out = 100 * tot / n_tot
    lo, hi = np.percentile(out, [2.5, 97.5])
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('runs', nargs='+')
    ap.add_argument('--reps', type=int, default=10000); ap.add_argument('--seed', type=int, default=20261109)
    a = ap.parse_args()
    df = MT.annotate(MT.load(a.runs), THR)
    df = df[df['eligible_m']].sort_values(['run', 'route', 'idx'])
    rows = []
    for mode, ms in MODES.items():
        for A, B in PAIRS:
            ra, rb = f'{A}_{mode}', f'{B}_{mode}'
            if ra not in set(df.run) or rb not in set(df.run): continue
            for group in ('doctoral', 'heldout'):
                ga = df[(df.run == ra) & (df.group == group)]; gb = df[(df.run == rb) & (df.group == group)]
                m = ga[['route', 'idx', 'file', 'correct']].merge(gb[['route', 'file', 'correct']], on=['route', 'file'],
                                                                  suffixes=('_a', '_b'), validate='1:1')
                assert len(m) == len(ga) == len(gb), (ra, rb, group, len(m), len(ga), len(gb))
                m = m.sort_values(['route', 'idx'])
                dA = {r: g['correct_a'].values for r, g in m.groupby('route')}
                dB = {r: g['correct_b'].values for r, g in m.groupby('route')}
                per_route = {r: 100 * (dA[r].mean() - dB[r].mean()) for r in dA}
                row = dict(mode=mode, a=REPS[A], b=REPS[B], group=group, n_frames=len(m), n_routes=len(dA),
                           succ_a=100 * m['correct_a'].mean(), succ_b=100 * m['correct_b'].mean(),
                           routes_a_better=sum(v > 0 for v in per_route.values()),
                           routes_b_better=sum(v < 0 for v in per_route.values()),
                           frames_a_only=int((m.correct_a & ~m.correct_b).sum()),
                           frames_b_only=int((~m.correct_a & m.correct_b).sum()))
                for L in (25, 10, 50, 'route'):
                    rng = np.random.default_rng(a.seed)
                    p, lo, hi = paired(dA, dB, L, a.reps, rng)
                    row['diff_pp'] = p; row[f'lo_{L}'] = lo; row[f'hi_{L}'] = hi
                rows.append(row)
                print(f"{mode:13s} {group:9s} {REPS[A]:>6s}-{REPS[B]:<6s} {row['diff_pp']:+6.1f}  "
                      f"L25 [{row['lo_25']:+.1f},{row['hi_25']:+.1f}]  L10 [{row['lo_10']:+.1f},{row['hi_10']:+.1f}]  "
                      f"L50 [{row['lo_50']:+.1f},{row['hi_50']:+.1f}]  route [{row['lo_route']:+.1f},{row['hi_route']:+.1f}]  "
                      f"routes {row['routes_a_better']}/{row['n_routes']}", flush=True)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(a.out + '.csv', index=False)
    json.dump(dict(reps=a.reps, seed=a.seed, block_main=25, rows=t.to_dict('records')), open(a.out + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
