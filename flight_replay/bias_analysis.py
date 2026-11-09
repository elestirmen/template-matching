"""Along-track / cross-track decomposition of accepted-fix errors per route (one run CSV)."""
import sys, numpy as np, pandas as pd
from rasterio.warp import transform
f = sys.argv[1]
d = pd.read_csv(f).drop_duplicates(['route', 'file'])
out = []
for r, g in d.groupby('route'):
    g = g.sort_values('idx')
    x, y = transform('EPSG:4326', 'EPSG:32636', list(g.lon_true), list(g.lat_true))
    g = g.assign(xt=x, yt=y)
    g['vx'] = np.gradient(g.xt.values); g['vy'] = np.gradient(g.yt.values)
    a = g[(g.resolved == 1) & g.lat_est.notna()].copy()
    if len(a) < 5: continue
    xe, ye = transform('EPSG:4326', 'EPSG:32636', list(a.lon_est), list(a.lat_est))
    a['dx'] = np.array(xe) - a.xt.values; a['dy'] = np.array(ye) - a.yt.values
    n = np.hypot(a.vx, a.vy) + 1e-9
    a['along'] = (a.dx * a.vx + a.dy * a.vy) / n; a['cross'] = (-a.dx * a.vy + a.dy * a.vx) / n
    ok = a[a.err_m <= 100]
    out.append(dict(route=r, camera=g.camera.iloc[0], n=len(ok), along_mean=ok.along.mean(), along_sd=ok.along.std(),
                    cross_mean=ok.cross.mean(), cross_sd=ok.cross.std(), east_mean=ok.dx.mean(), north_mean=ok.dy.mean(),
                    resid_after_bias=np.median(np.hypot(ok.along - ok.along.mean(), ok.cross - ok.cross.mean()))))
print(pd.DataFrame(out).round(1).to_string(index=False))
pd.DataFrame(out).round(2).to_csv(sys.argv[2] if len(sys.argv) > 2 else 'bias.csv', index=False)
