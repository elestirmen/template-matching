"""Camera boresight bias in the body frame (forward/right relative to the recorded yaw), estimated per camera
with leave-one-route-out, and its effect on the error of accepted fixes. Uses GPS only on the calibration routes.
Usage: python bias_loo.py RUN.csv OUT.csv"""
import sys, numpy as np, pandas as pd
from rasterio.warp import transform
d = pd.read_csv(sys.argv[1]).drop_duplicates(['route', 'file'])
a = d[(d.resolved == 1) & d.lat_est.notna()].copy()
xt, yt = transform('EPSG:4326', 'EPSG:32636', list(a.lon_true), list(a.lat_true))
xe, ye = transform('EPSG:4326', 'EPSG:32636', list(a.lon_est), list(a.lat_est))
a['dE'] = np.array(xe) - np.array(xt); a['dN'] = np.array(ye) - np.array(yt)
yaw = np.radians(a['yaw'].values)
fE, fN = np.sin(yaw), np.cos(yaw); rE, rN = np.cos(yaw), -np.sin(yaw)
a['fwd'] = a.dE * fE + a.dN * fN; a['right'] = a.dE * rE + a.dN * rN
a['doctoral'] = ~a.route.str.endswith('heldout')
rows = []
for r, g in a.groupby('route'):
    cal = a[(a.camera == g.camera.iloc[0]) & (a.route != r) & a.doctoral & (a.err_m <= 100)]
    bf, br = cal.fwd.median(), cal.right.median()
    y = np.radians(g['yaw'].values)
    cE = bf * np.sin(y) + br * np.cos(y); cN = bf * np.cos(y) - br * np.sin(y)
    e_corr = np.hypot(g.dE - cE, g.dN - cN)
    rows.append(dict(route=r, camera=g.camera.iloc[0], accepted=len(g), bias_fwd=bf, bias_right=br,
                     prec_before=100 * (g.err_m <= 30).mean(), prec_after=100 * (e_corr <= 30).mean(),
                     median_before=g.err_m.median(), median_after=np.median(e_corr)))
t = pd.DataFrame(rows).round(1); print(t.to_string(index=False)); t.to_csv(sys.argv[2], index=False)
for grp, sel in (('doctoral', ~t.route.str.endswith('heldout')), ('additional', t.route.str.endswith('heldout'))):
    s = t[sel]; w = s.accepted
    print(grp, 'accepted', int(w.sum()), 'prec before %.1f after %.1f' % ((s.prec_before * w).sum() / w.sum(), (s.prec_after * w).sum() / w.sum()))
print('camera biases (all doctoral routes):')
cal = a[a.doctoral & (a.err_m <= 100)]
print(cal.groupby('camera')[['fwd', 'right']].median().round(1))
