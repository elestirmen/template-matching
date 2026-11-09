"""Scale calibration: for a few frames, find the frame scale factor that maximises the NCC score
at the TRUE (GPS) location, and express it as an equivalent height-above-ground.
Usage: python calib_scale.py FRAMES_DIR MAP DEM MODEL.onnx OUT.csv [--every N] [--dem-offset 26]
"""
import os, sys, argparse, csv, math, glob
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_flights import Raster, rotate_inscribed, CAMERAS, MAP_GSD, TILE, BORDER, TPL, K_L1D, D_L1D
from cae_infer import CAE
import piexif, re

ap = argparse.ArgumentParser()
ap.add_argument('frames'); ap.add_argument('map'); ap.add_argument('dem'); ap.add_argument('model'); ap.add_argument('out')
ap.add_argument('--every', type=int, default=20); ap.add_argument('--dem-offset', type=float, default=26.0)
ap.add_argument('--factors', default='0.80,0.84,0.88,0.92,0.96,1.00,1.04,1.08,1.12,1.16,1.20,1.25')
ap.add_argument('--route', default='')
a = ap.parse_args()
factors = [float(x) for x in a.factors.split(',')]
mp = Raster(a.map); dem = Raster(a.dem); cae = CAE(a.model) if a.model else None
files = sorted(glob.glob(os.path.join(a.frames, '*.JPG')))[::a.every]
rows = []
for fp in files:
    ex = piexif.load(fp); g = ex['GPS']; e1 = ex['Exif']; e0 = ex['0th']
    rat = lambda v: v[0] / v[1]
    dms = lambda v, ref: (rat(v[0]) + rat(v[1]) / 60 + rat(v[2]) / 3600) * (-1 if ref in (b'S', b'W') else 1)
    lat = dms(g[piexif.GPSIFD.GPSLatitude], g[piexif.GPSIFD.GPSLatitudeRef]); lon = dms(g[piexif.GPSIFD.GPSLongitude], g[piexif.GPSIFD.GPSLongitudeRef])
    alt = rat(g[piexif.GPSIFD.GPSAltitude]); focal = rat(e1[piexif.ExifIFD.FocalLength]); cam = e0[piexif.ImageIFD.Model].decode().strip('\x00 ')
    head = open(fp, 'rb').read(200000); m = re.search(rb'drone-dji:FlightYawDegree="([^"]*)"', head)
    yaw = float(m.group(1)) if m else float('nan')
    C = CAMERAS[cam]; dv = dem.sample(lat, lon)
    if not np.isfinite(dv): continue
    h0 = alt - dv + a.dem_offset
    img = cv2.imread(fp, 0)
    if cam == 'L1D-20c':
        m1, m2 = cv2.initUndistortRectifyMap(K_L1D, D_L1D, None, K_L1D, (img.shape[1], img.shape[0]), cv2.CV_16SC2); img = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
    rot = rotate_inscribed(img, -yaw)
    tc, tr = mp.ll_to_px(lat, lon)
    win, c0, r0 = mp.read(tc - 400, tr - 400, 800, 800)
    best = None
    for f in factors:
        h = h0 * f; gsd = C['sensor_w'] * h / (focal * C['img_w']); s = gsd / MAP_GSD
        nh, nw = int(rot.shape[0] * s), int(rot.shape[1] * s)
        if min(nh, nw) < TILE: continue
        norm = cv2.resize(rot, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        cx, cy = nw // 2, nh // 2
        crop = norm[cy - 272: cy + 272, cx - 272: cx + 272]
        tpl = (cae.run([crop])[0] if cae else crop)[BORDER:TILE - BORDER, BORDER:TILE - BORDER]
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(res)
        ex_, ey_ = c0 + loc[0] + TPL / 2, r0 + loc[1] + TPL / 2
        err = math.hypot(ex_ - tc, ey_ - tr) * MAP_GSD
        rows.append(dict(route=a.route or os.path.basename(a.frames), file=os.path.basename(fp), camera=cam, h0=round(h0, 1), factor=f, h=round(h, 1), scale=round(s, 4), score=round(mx, 4), err_m=round(err, 1)))
        if best is None or mx > best[1]: best = (f, mx, err)
    print(os.path.basename(fp), cam, 'h0=%.0f' % h0, 'best factor %.2f score %.3f err %.0f m' % best, flush=True)
with open(a.out, 'a', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
    if fo.tell() == 0: w.writeheader()
    w.writerows(rows)
