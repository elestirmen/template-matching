#!/usr/bin/env python3
"""Low-texture probe for the cross-source benchmark (Reviewer 2, item 5).

The benchmark rejects query tiles with grey-level standard deviation < 12 or entropy < 4 bits
(and tiles with > 20 % near-black pixels, i.e. missing data). This probe measures how the
representations behave on exactly those low-texture tiles, against regular tiles drawn from the
same grid with the same matcher.

Design
  * Regular 500 m grid over the common extent of the district Google / Bing mosaics.
  * Each grid point -> 544 x 544 query tile on the query mosaic; gate statistics computed as in
    the benchmark. 'low' = std < 12 or entropy < 4 (and dark fraction <= 0.2);
    'regular' = passes the gate. Dark (missing-data) tiles are excluded from both strata.
  * All 'low' points and a random sample (seed 42) of up to N_REG 'regular' points are matched.
  * Search: window of radius ROI m around the true centre on the reference mosaic, coarse-to-fine
    NCC (same code as the flight replay), success = error <= 25 m.
  * Representations: untransformed greyscale, gradient magnitude (Sobel), CAE f48-k4-s1 ckpt 1850.
  * Both directions (Google -> Bing, Bing -> Google), clean tiles only.
Usage: python texture_probe.py [--roi 1000] [--n-reg 300]
"""
import os, sys, csv, math, argparse, subprocess, time
import numpy as np, cv2, rasterio
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_flights import match_in_window
from cae_infer import make_transform

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BENCH = os.path.normpath(os.path.join(ROOT, '..', 'benchmark', 'urgup_cevresi_harita'))
RASTERS = {'google': os.environ.get('TP_GOOGLE', os.path.join(BENCH, 'urgup_cevresi_gmap_utm_30cm.tif')),
           'bing': os.environ.get('TP_BING', os.path.join(BENCH, 'urgup_cevresi_bingmap_utm_30cm.tif'))}
import models_registry as MR
CAE_ONNX = os.path.normpath(os.path.join(ROOT, MR.model_path('f48k4s1')))  # .h5 (TensorFlow) or .onnx
OUTDIR = os.environ.get('TP_OUT', os.path.join(HERE, 'texture_probe'))
T, B, TPL = 544, 16, 512


def entropy_u8(g):
    h = np.bincount(g.ravel(), minlength=256).astype(np.float64); p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def gray_window(ds, c0, r0, w, h):
    a = ds.read(window=Window(int(c0), int(r0), int(w), int(h)))
    return cv2.cvtColor(np.transpose(a[:3], (1, 2, 0)), cv2.COLOR_RGB2GRAY) if a.shape[0] >= 3 else a[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roi', type=float, default=1000.0)
    ap.add_argument('--n-reg', type=int, default=300)
    ap.add_argument('--grid', type=float, default=500.0)
    ap.add_argument('--margin', type=float, default=1500.0)
    ap.add_argument('--eq', type=int, default=0, help='histogram equalisation before the transform (0 = benchmark protocol)')
    ap.add_argument('--out', default='', help='output folder (default: OUTDIR)')
    a = ap.parse_args()
    global OUTDIR
    if a.out: OUTDIR = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(OUTDIR, exist_ok=True)
    # 1) structural reference rasters (resumable)
    maps = {}
    for name, src in RASTERS.items():
        for rep, model in (('cae', CAE_ONNX), ('sobel', 'sobel')):
            out = os.path.join(OUTDIR, f'district_{name}_{rep}.tif')
            subprocess.call([sys.executable, os.path.join(HERE, 'transform_map.py'), src, out, model, '--eq', str(a.eq)], cwd=ROOT)
            with rasterio.open(out) as ds: nby = -(-ds.height // 512)
            rd = out + '.rows_done'
            if not os.path.exists(rd) or len(set(open(rd).read().split())) < nby:
                print('ABORT: incomplete map', out); sys.exit(1)
            maps[(name, rep)] = out
    trans = {'cae': make_transform(CAE_ONNX), 'sobel': make_transform('sobel')}
    res_path = os.path.join(OUTDIR, 'texture_probe_results.csv')
    done = set()
    if os.path.exists(res_path):
        with open(res_path, newline='') as f:
            for r in csv.DictReader(f): done.add((r['direction'], r['point_id'], r['representation']))
    fo = open(res_path, 'a', newline='')
    wr = csv.DictWriter(fo, fieldnames=['direction', 'point_id', 'stratum', 'x', 'y', 'std', 'entropy', 'dark', 'representation',
                                        'error_m', 'score', 'success_25m'])
    if not done: wr.writeheader()
    rng = np.random.default_rng(42)
    for qname, rname in (('google', 'bing'), ('bing', 'google')):
        qds = rasterio.open(RASTERS[qname]); rds = rasterio.open(RASTERS[rname])
        rmaps = {rep: rasterio.open(maps[(rname, rep)]) for rep in ('cae', 'sobel')}
        b = qds.bounds; res = qds.res[0]
        pts = []
        k = 0
        for y in np.arange(b.bottom + a.margin, b.top - a.margin, a.grid):
            for x in np.arange(b.left + a.margin, b.right - a.margin, a.grid):
                k += 1
                c, r = ~qds.transform * (x, y); c0, r0 = int(c - T / 2), int(r - T / 2)
                if c0 < 0 or r0 < 0 or c0 + T > qds.width or r0 + T > qds.height: continue
                g = gray_window(qds, c0, r0, T, T)
                s, e, d = float(np.std(g)), entropy_u8(g), float(np.mean(g <= 5))
                if d > 0.2: continue
                stratum = 'low' if (s < 12 or e < 4) else 'regular'
                pts.append(dict(point_id=f'P{k:05d}', x=c0 + T / 2, y=r0 + T / 2, std=s, entropy=e, dark=d, stratum=stratum, tile=g))
        low = [p for p in pts if p['stratum'] == 'low']
        reg = [p for p in pts if p['stratum'] == 'regular']
        reg = [reg[i] for i in sorted(rng.choice(len(reg), min(a.n_reg, len(reg)), replace=False))]
        print(f'{qname}->{rname}: grid points {len(pts)}, low {len(low)}, regular sampled {len(reg)}', flush=True)
        R = int(a.roi / res)
        for n_p, p in enumerate(low + reg):
            if n_p % 25 == 0:
                print(f'  {qname}->{rname}: point {n_p}/{len(low) + len(reg)}', flush=True)
            for rep in ('gray', 'sobel', 'cae'):
                if (f'{qname}_to_{rname}', p['point_id'], rep) in done: continue
                tile = p['tile']
                if rep == 'gray':
                    tpl = tile[B:T - B, B:T - B]; src = rds; is_rgb = True
                else:
                    tpl = trans[rep].run([tile], equalize=bool(a.eq))[0][B:T - B, B:T - B]; src = rmaps[rep]; is_rgb = False
                # query centre in reference pixel coordinates (same CRS and grid spacing)
                qx, qy = qds.transform * (p['x'], p['y'])
                cc, cr = ~src.transform * (qx, qy)
                c0, r0 = max(0, int(cc - R - TPL / 2)), max(0, int(cr - R - TPL / 2))
                w = min(src.width - c0, 2 * R + TPL); h = min(src.height - r0, 2 * R + TPL)
                win = gray_window(src, c0, r0, w, h) if is_rgb else src.read(1, window=Window(c0, r0, w, h))
                m = match_in_window(win, tpl)
                if m is None:
                    err, sc = float('nan'), float('nan')
                else:
                    ex, ey = c0 + m[0] + TPL / 2, r0 + m[1] + TPL / 2
                    err = math.hypot(ex - cc, ey - cr) * res; sc = m[2]
                wr.writerow(dict(direction=f'{qname}_to_{rname}', point_id=p['point_id'], stratum=p['stratum'], x=round(qx, 1), y=round(qy, 1),
                                 std=round(p['std'], 2), entropy=round(p['entropy'], 3), dark=round(p['dark'], 4), representation=rep,
                                 error_m=round(err, 2) if err == err else '', score=round(sc, 4) if sc == sc else '',
                                 success_25m=int(err == err and err <= 25.0)))
            fo.flush()
    fo.close()
    # summary (csv module only)
    agg = {}
    with open(res_path, newline='') as f:
        for r in csv.DictReader(f):
            k = (r['direction'], r['stratum'], r['representation'])
            n, s_ = agg.get(k, (0, 0)); agg[k] = (n + 1, s_ + int(r['success_25m']))
    with open(os.path.join(OUTDIR, 'texture_probe_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['direction', 'stratum', 'representation', 'n', 'success_pct'])
        for k in sorted(agg):
            n, s_ = agg[k]; w.writerow(list(k) + [n, round(100.0 * s_ / n, 1)]); print(k, n, round(100.0 * s_ / n, 1))

if __name__ == '__main__':
    main()
