"""Fig. 6: GPS track (green) and accepted visual fixes (yellow; red = wrong) over the reference mosaic.
Usage: python make_fig6.py RUN.csv ROUTE MOSAIC.tif OUT.png [--pad 400] [--scale 4]
"""
import sys, argparse, numpy as np, cv2, rasterio, pandas as pd
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform

ap = argparse.ArgumentParser()
ap.add_argument('csv'); ap.add_argument('route'); ap.add_argument('mosaic'); ap.add_argument('out')
ap.add_argument('--pad', type=int, default=500); ap.add_argument('--scale', type=int, default=4)
ap.add_argument('--title', default='')
a = ap.parse_args()
d = pd.read_csv(a.csv); d = d[d.route == a.route]
ds = rasterio.open(a.mosaic)
def px(lat, lon):
    xs, ys = rio_transform('EPSG:4326', ds.crs, list(lon), list(lat))
    cols, rows = [], []
    for x, y in zip(xs, ys):
        c, r = ~ds.transform * (x, y); cols.append(c); rows.append(r)
    return np.array(cols), np.array(rows)
tc, tr = px(d.lat_true, d.lon_true)
acc = d[d.resolved == 1]
ec, er = px(acc.lat_est, acc.lon_est)
c0, c1 = int(max(0, min(tc.min(), ec.min()) - a.pad)), int(min(ds.width, max(tc.max(), ec.max()) + a.pad))
r0, r1 = int(max(0, min(tr.min(), er.min()) - a.pad)), int(min(ds.height, max(tr.max(), er.max()) + a.pad))
s = a.scale
img = ds.read(window=Window(c0, r0, c1 - c0, r1 - r0), out_shape=(ds.count, (r1 - r0) // s, (c1 - c0) // s))
if img.shape[0] >= 3:
    im = np.transpose(img[:3], (1, 2, 0))[:, :, ::-1].copy()  # BGR for cv2
else:
    im = cv2.cvtColor(img[0], cv2.COLOR_GRAY2BGR)
pts = np.stack([(tc - c0) / s, (tr - r0) / s], 1).astype(np.int32)
cv2.polylines(im, [pts], False, (0, 200, 0), 3)
for (x, y), e in zip(zip((ec - c0) / s, (er - r0) / s), acc.err_m):
    col = (0, 230, 255) if e <= 30 else (0, 0, 255)
    cv2.circle(im, (int(x), int(y)), 5, col, -1)
# scale bar 200 m
L = int(200 / 0.2986 / s)
cv2.rectangle(im, (20, im.shape[0] - 40), (20 + L, im.shape[0] - 30), (255, 255, 255), -1)
cv2.putText(im, '200 m', (20, im.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
if a.title: cv2.putText(im, a.title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.imwrite(a.out, im)
print('wrote', a.out, im.shape, 'accepted', len(acc), 'wrong', int((acc.err_m > 30).sum()))
