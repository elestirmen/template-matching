#!/usr/bin/env python3
"""
Flight replay for cross-autoencoder / NCC UAV localization (clean re-evaluation).

One script, one per-frame record format, for every route, model and mode.
Everything a table in the article needs is recomputed from the CSV this writes.

Pipeline per frame
  1. EXIF/XMP: GPS lat/lon (reference only), GPS altitude, barometric RelativeAltitude,
     FlightYawDegree, focal length, camera model.
  2. Height above ground H = altitude - DEM(position) + DEM_OFFSET
        mode 'aided'  : altitude = GPS altitude of the frame, DEM sampled at the GPS position
        mode 'causal' : altitude = (GPS altitude - RelativeAltitude) of the FIRST frame
                        + RelativeAltitude of this frame (barometric); DEM sampled at the
                        previous visual estimate (first frame: initial GPS position)
  3. Grayscale, optional undistortion (L1D-20c), rotation to north by -yaw, largest
     inscribed rectangle, resampling to the map GSD (0.2986 m/px).
  4. Three 544x544 crops: centre, upper-left and lower-right (displaced by d = min(200, margin) px
     along both axes).  Each crop -> CAE (ONNX) -> central 512x512 template.
     Grayscale baseline: the untransformed crops, central 512x512.
  5. NCC (TM_CCOEFF_NORMED) of each template inside the search window, coarse-to-fine
     (1/4 scale, top-5 candidates, full-resolution refinement).
  6. Consensus ('strict', default): each matched crop implies a position of the frame centre
     (its peak corrected by the known crop displacement); crops whose implied centres agree
     within TOL px form the consensus and their mean is the position. A peak on the window
     border is discarded. Fewer than two agreeing crops -> UNRESOLVED (no position output).
     ('loose' reproduces the thesis rule: intersection of 512 px candidate boxes.)
  7. Search window: 'oracle' mode centres a fixed window on the frame's own GPS position
     (Simulation-3 protocol, no gate); 'track' modes centre it on a predicted position: the
     last accepted visual fix propagated frame to frame by a visual-odometry prior (phase
     correlation between consecutive north-up, map-scaled frames; the last displacement is
     reused when the correlation is weak). The first frame starts at the initial position (GPS
     at initialisation). Nominal window WIN px, enlarged by +GROW px per frame without an
     accepted fix up to WIN_MAX, reset after a fix. A consensus position is accepted only if it
     lies within GATE + GATE_GROWTH x (frames since the last fix) px of the prediction (capped at
     half the window); otherwise it is rejected (reason 'gate') and counts as unresolved. No GNSS
     information is used after the first frame in 'track' modes except where mode 'aided' says
     so (altitude/DEM only).

Per-frame CSV columns are documented in write_header().
"""
import os, sys, csv, time, math, argparse, json


def _supervise():
    """Run this replay in a child process and retry it (the replay is resumable) when the child dies or prints
    nothing for 480 s. A sticky CUDA fault in the child's output (see gpu_fault.py) is NOT retried: the supervisor
    returns GPU_FAULT_RC and runs no queued jobs, because restarting on a faulted GPU ended in a hung NVIDIA driver
    on 2026-11-09; resume after a reboot. Otherwise, afterwards run the queued re-runs that write the
    same CSV (flight_replay/rerun_queue/<csv>.txt, one JSON argument list per line), i.e. route jobs left incomplete
    by an earlier crash; a queue file is only ever read by the jobs of its own CSV, which run one at a time."""
    import subprocess, threading
    env = dict(os.environ, VN_REPLAY_CHILD='1', PYTHONUNBUFFERED='1')
    me = os.path.abspath(__file__)
    sys.path.insert(0, os.path.dirname(me))
    from gpu_fault import GPU_FAULT_RC, is_gpu_fault

    def run_once(args):
        rc = 1
        for attempt in range(1, 4):
            p = subprocess.Popen([sys.executable, me] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                 text=True, encoding='utf-8', errors='replace', bufsize=1)
            last = [time.time()]
            fault_t = [None]  # time the child first reported a sticky CUDA fault

            def reader():
                for line in p.stdout:
                    last[0] = time.time()
                    if fault_t[0] is None and is_gpu_fault(line):
                        fault_t[0] = time.time()
                    try:
                        sys.stdout.write(line); sys.stdout.flush()
                    except Exception:
                        pass
            t = threading.Thread(target=reader, daemon=True); t.start()
            while p.poll() is None:
                time.sleep(5)
                if fault_t[0] is not None and time.time() - fault_t[0] > 60:
                    p.kill(); print('[replay] child still running 60 s after a CUDA fault; killed', flush=True); break
                if time.time() - last[0] > 480:
                    p.kill(); print('[replay] no output for 480 s; child killed', flush=True); break
            rc = p.wait(); t.join(timeout=10)
            if fault_t[0] is not None:
                print(f'[replay] CUDA fault on the GPU (exit code {rc}); NOT retrying. '
                      'Reboot, then re-run to resume.', flush=True)
                return GPU_FAULT_RC
            if rc == 0: return 0
            print(f'[replay] attempt {attempt} ended with exit code {rc}; ' + ('retrying (resumable)' if attempt < 3 else 'giving up'), flush=True)
        return rc

    args = sys.argv[1:]
    rc = run_once(args)
    if rc == GPU_FAULT_RC:
        return rc
    try:
        out = args[args.index('--out') + 1]
        qdir = os.path.join(os.path.dirname(me), 'rerun_queue')
        qf = os.path.join(qdir, out.replace('/', '_').replace('\\', '_') + '.txt')
        while os.path.exists(qf):
            lines = [l for l in open(qf, encoding='utf-8').read().splitlines() if l.strip()]
            if not lines:
                os.remove(qf); break
            job = json.loads(lines[0])
            with open(qf, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines[1:]) + ('\n' if len(lines) > 1 else ''))
            print('[replay] queued re-run:', ' '.join(job), flush=True)
            if run_once(job) == GPU_FAULT_RC:
                with open(qf, 'w', encoding='utf-8') as f:  # put the job back for the run after the reboot
                    f.write('\n'.join(lines) + '\n')
                return GPU_FAULT_RC
    except Exception as e:
        print('[replay] queue error:', e, flush=True)
    return rc


if __name__ == '__main__' and os.environ.get('VN_REPLAY_CHILD') != '1':
    sys.exit(_supervise())

import numpy as np, cv2, rasterio
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cae_infer import CAE, equalize_pil, make_transform

MAP_GSD = 0.2986          # m/px of the reference rasters (EPSG:32636)
TILE, BORDER, TPL = 544, 16, 512
CAMERAS = {  # sensor width (mm), native image width (px)
    'L1D-20c': dict(sensor_w=13.2, img_w=5472),
    'FC2204': dict(sensor_w=6.17, img_w=4000),
}
K_L1D = np.array([[4479.94, -1.917, 2727.48], [0, 4374.51, 1811.92], [0, 0, 1]], dtype=np.float64)
D_L1D = np.array([-0.0196948, 0.00652119, 0.000237354, 0.0, 0.0909295], dtype=np.float64)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1; dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class Raster:
    """Random-access reader with lat/lon <-> pixel conversion."""
    def __init__(self, path, band=1):
        self.ds = rasterio.open(path); self.band = band
        self.w, self.h = self.ds.width, self.ds.height
        self.crs = self.ds.crs

    def ll_to_px(self, lat, lon):
        xs, ys = rio_transform('EPSG:4326', self.crs, [lon], [lat])
        c, r = ~self.ds.transform * (xs[0], ys[0])
        return float(c), float(r)

    def px_to_ll(self, c, r):
        x, y = self.ds.transform * (c, r)
        lons, lats = rio_transform(self.crs, 'EPSG:4326', [x], [y])
        return float(lats[0]), float(lons[0])

    def read(self, c0, r0, w, h, gray=True):
        c0, r0 = int(c0), int(r0)
        c0 = max(0, min(c0, self.w - 1)); r0 = max(0, min(r0, self.h - 1))
        w = min(w, self.w - c0); h = min(h, self.h - r0)
        arr = self.ds.read(window=Window(c0, r0, w, h))
        if arr.shape[0] >= 3 and gray:
            arr = cv2.cvtColor(np.transpose(arr[:3], (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[0]
        return arr, c0, r0

    def sample(self, lat, lon):
        c, r = self.ll_to_px(lat, lon)
        if not (0 <= c < self.w and 0 <= r < self.h): return float('nan')
        v = self.ds.read(self.band, window=Window(int(c), int(r), 1, 1))
        return float(v[0, 0])


def rotate_inscribed(gray, angle_deg):
    """Rotate by angle (deg, CCW positive as in cv2) and keep the largest axis-aligned inscribed rectangle."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    rot = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR)
    # largest inscribed rectangle (Coneybeare / rotated rect)
    a = math.radians(angle_deg) % math.pi
    if a > math.pi / 2: a = math.pi - a
    sa, ca = math.sin(a), math.cos(a)
    wl, ws = (w, h) if w >= h else (h, w)
    if ws <= 2 * sa * ca * wl or abs(sa - ca) < 1e-10:
        x = 0.5 * ws
        rw, rh = (x / sa, x / ca) if w >= h else (x / ca, x / sa)
    else:
        c2 = ca * ca - sa * sa
        rw, rh = (w * ca - h * sa) / c2, (h * ca - w * sa) / c2
    rw, rh = int(rw), int(rh)
    cx, cy = nw // 2, nh // 2
    return rot[cy - rh // 2: cy + rh // 2, cx - rw // 2: cx + rw // 2]


def match_in_window(win, tpl, coarse=4, topk=5, pad=96):
    """Coarse-to-fine NCC. Returns (col, row, score) of the template's top-left in window coordinates."""
    H, W = win.shape; h, w = tpl.shape
    if H < h or W < w: return None
    if coarse > 1 and H > 4 * h:
        ws = cv2.resize(win, (W // coarse, H // coarse), interpolation=cv2.INTER_AREA)
        ts = cv2.resize(tpl, (w // coarse, h // coarse), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(ws, ts, cv2.TM_CCOEFF_NORMED)
        cands = []
        r = res.copy(); nms = max(1, (h // coarse) // 2)
        for _ in range(topk):
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx <= -1: break
            cands.append((loc[0] * coarse, loc[1] * coarse))
            x0, y0 = max(0, loc[0] - nms), max(0, loc[1] - nms)
            r[y0: loc[1] + nms + 1, x0: loc[0] + nms + 1] = -1
        best = None
        for (cx, cy) in cands:
            x0, y0 = max(0, cx - pad), max(0, cy - pad)
            x1, y1 = min(W, cx + w + pad), min(H, cy + h + pad)
            sub = win[y0:y1, x0:x1]
            if sub.shape[0] < h or sub.shape[1] < w: continue
            rr = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(rr)
            if best is None or mx > best[2]: best = (x0 + loc[0], y0 + loc[1], float(mx))
        return best
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return (loc[0], loc[1], float(mx))


def inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    if x1 <= x0 or y1 <= y0: return None
    return (x0, y0, x1 - x0, y1 - y0)


def consensus_loose(boxes):
    """Thesis rule. boxes: list of (x, y, w, h) or None. Returns (centre_x, centre_y, n_agree) or None."""
    a, b, c = boxes
    if a and b and c:
        ab, bc, ac = inter(a, b), inter(b, c), inter(a, c)
        if ab and bc and ac:
            abc = inter(ab, bc)
            if abc: return (abc[0] + abc[2] / 2, abc[1] + abc[3] / 2, 3)
    for p, q in ((a, b), (b, c), (a, c)):
        if p and q:
            pq = inter(p, q)
            if pq: return (pq[0] + pq[2] / 2, pq[1] + pq[3] / 2, 2)
    return None


def consensus_strict(peaks, d, tol):
    """Geometric consistency. peaks: top-left (x, y) of the matched 512 px template for the
    upper-left, centre and lower-right crops (or None). Each peak implies a position of the frame
    centre; crops whose implied centres agree within tol px form the consensus. Returns
    (centre_x, centre_y, n_agree) or None (no two crops agree)."""
    offs = [(+d, +d), (0, 0), (-d, -d)]
    imp = [None if p is None else (p[0] + TPL / 2 + o[0], p[1] + TPL / 2 + o[1]) for p, o in zip(peaks, offs)]
    best = None
    for i in range(3):
        if imp[i] is None: continue
        group = [j for j in range(3) if imp[j] is not None and math.hypot(imp[j][0] - imp[i][0], imp[j][1] - imp[i][1]) <= tol]
        if len(group) >= 2 and (best is None or len(group) > len(best)):
            best = group
    if best is None: return None
    xs = [imp[j][0] for j in best]; ys = [imp[j][1] for j in best]
    return (sum(xs) / len(xs), sum(ys) / len(ys), len(best))


FIELDS = ['run', 'model', 'mode', 'route', 'idx', 'file', 'datetime', 'camera', 'eligible', 'reason',
          'lat_true', 'lon_true', 'gps_alt', 'rel_alt', 'yaw', 'alt_used', 'dem_at', 'h_agl', 'scale', 'disp_px',
          'win_cx', 'win_cy', 'win_px', 'win_center_src', 'vo_dx', 'vo_dy', 'vo_resp', 'vo_src', 'border', 'gate_px', 'pred_dist_px',
          'resolved', 'n_agree', 'score_ul', 'score_c', 'score_lr',
          'est_col', 'est_row', 'lat_est', 'lon_est', 'err_m', 'sec']
# reason: '' = accepted fix; 'no_consensus' = crops disagree (unresolved); 'gate' = consensus found but
# inconsistent with the motion prediction (rejected); other values = frame not eligible.



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True, help='frame directory')
    ap.add_argument('--route', required=True, help='route label written to the CSV')
    ap.add_argument('--map', required=True, help='reference raster (structural or RGB mosaic)')
    ap.add_argument('--dem', required=True, help='DEM raster (any CRS, tiled)')
    ap.add_argument('--model', default='', help="ONNX model; '' = grayscale baseline; 'sobel' = gradient-magnitude baseline")
    ap.add_argument('--mode', choices=['oracle', 'track_aided', 'track_causal'], required=True)
    ap.add_argument('--out', required=True, help='per-frame CSV (appended, resumable)')
    ap.add_argument('--run', default='', help='run label')
    ap.add_argument('--win', type=int, default=2000); ap.add_argument('--win-max', type=int, default=6000)
    ap.add_argument('--grow', type=int, default=500)
    ap.add_argument('--oracle-win', type=int, default=10000)
    ap.add_argument('--dem-offset', type=float, default=26.0)
    ap.add_argument('--yaw-offset', type=float, default=0.0)
    ap.add_argument('--yaw-source', choices=['gimbal', 'flight'], default='flight', help='camera heading: gimbal yaw (camera) or flight yaw (body)')
    ap.add_argument('--yaw-sign', type=float, default=-1.0, help='rotation angle = yaw_sign * yaw + yaw_offset (cv2 convention, +=CCW); -1 verified on flight data')
    ap.add_argument('--alt-offset', type=float, default=0.0, help='sensitivity: add to altitude (m)')
    ap.add_argument('--alt-scale', type=float, default=1.0, help='sensitivity: multiply height above ground')
    ap.add_argument('--init-offset-m', type=float, default=0.0, help='sensitivity: displace the initial (first-frame) position north-east by this many metres')
    ap.add_argument('--eq', type=int, default=1, help='PIL-style histogram equalisation before the CAE')
    ap.add_argument('--consensus', choices=['strict', 'loose'], default='strict')
    ap.add_argument('--tol', type=float, default=48.0, help='strict consensus: agreement tolerance (px)')
    ap.add_argument('--gate', type=float, default=300.0, help='track modes: acceptance gate around the predicted position (px); 0 = off')
    ap.add_argument('--gate-growth', type=float, default=100.0, help='track modes: gate growth per frame without an accepted fix (px)')
    ap.add_argument('--vo', type=int, default=1, help='track modes: visual-odometry prior (phase correlation between consecutive frames)')
    ap.add_argument('--vo-size', type=int, default=1088, help='side of the central region used for the VO prior (px at map GSD)')
    ap.add_argument('--vo-min-resp', type=float, default=0.05)
    ap.add_argument('--undistort', type=int, default=1)
    ap.add_argument('--budget', type=float, default=1e9, help='seconds; stop before exceeding (resumable)')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--state', default='', help='json file holding tracking state for resumption')
    a = ap.parse_args()

    import piexif, re, glob
    _skip = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SKIP_EQ1_MODELS.txt')
    if a.eq == 1 and a.model and os.path.exists(_skip):  # superseded runs with equalisation (see run_all.py, VN_EQ)
        if any(t.strip() and t.strip() in a.model for t in open(_skip).read().splitlines()):
            print('skipped (listed in SKIP_EQ1_MODELS.txt):', a.model, a.route, flush=True); return
    # case-insensitive file systems (Windows) return the same file for '*.JPG' and '*.jpg': deduplicate
    _seen = {}
    for _f in glob.glob(os.path.join(a.frames, '*.JPG')) + glob.glob(os.path.join(a.frames, '*.jpg')):
        _seen.setdefault(os.path.normcase(os.path.abspath(_f)), _f)
    def _shot_time(p):  # capture order: EXIF DateTimeOriginal (DJI file numbers wrap at 0999 -> 0001)
        try:
            return piexif.load(p)['Exif'].get(piexif.ExifIFD.DateTimeOriginal, b'').decode('ascii', 'ignore')
        except Exception:
            return ''
    files = sorted(_seen.values(), key=lambda p: (_shot_time(p), os.path.basename(p).lower()))
    if a.max_frames: files = files[:a.max_frames]
    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline='') as f:
            for r in csv.DictReader(f):
                if r['run'] == a.run and r['route'] == a.route: done.add(r['file'])
    newfile = not os.path.exists(a.out)
    fout = open(a.out, 'a', newline=''); wr = csv.DictWriter(fout, fieldnames=FIELDS)
    if newfile: wr.writeheader()
    state_path = a.state or (a.out + f'.{a.run}.{a.route}.state.json')
    st = json.load(open(state_path)) if os.path.exists(state_path) else {}

    rows_done = a.map + '.rows_done'
    if a.model:  # a structural map must be complete (sidecar written by transform_map.py)
        import rasterio as _rio
        with _rio.open(a.map) as _ds: _nby = -(-_ds.height // 512)
        _n = len(set(open(rows_done).read().split())) if os.path.exists(rows_done) else 0
        if _n < _nby:
            raise SystemExit(f'ERROR: structural map incomplete ({_n}/{_nby} rows): {a.map}. Run transform_map.py first.')
    mp = Raster(a.map); dem = Raster(a.dem)
    cae = make_transform(a.model)
    maps_undist = {}
    vo_prev = {}
    vo_win = cv2.createHanningWindow((256, 256), cv2.CV_32F)
    t_start = time.time()

    def read_meta(fp):
        ex = piexif.load(fp); g = ex['GPS']; e1 = ex['Exif']; e0 = ex['0th']
        def rat(v): return v[0] / v[1]
        def dms(v, ref):
            d = rat(v[0]) + rat(v[1]) / 60 + rat(v[2]) / 3600
            return -d if ref in (b'S', b'W') else d
        lat = dms(g[piexif.GPSIFD.GPSLatitude], g[piexif.GPSIFD.GPSLatitudeRef])
        lon = dms(g[piexif.GPSIFD.GPSLongitude], g[piexif.GPSIFD.GPSLongitudeRef])
        alt = rat(g[piexif.GPSIFD.GPSAltitude])
        with open(fp, 'rb') as f: head = f.read(200000)
        m = re.search(rb'<x:xmpmeta.*?</x:xmpmeta>', head, re.S); xmp = m.group(0).decode('utf-8', 'ignore') if m else ''
        def xv(k):
            m = re.search(r'drone-dji:%s="([^"]*)"' % k, xmp) or re.search(r'<drone-dji:%s>([^<]*)</drone-dji:%s>' % (k, k), xmp)
            return float(m.group(1)) if m else float('nan')
        return dict(lat=lat, lon=lon, gps_alt=alt, rel_alt=xv('RelativeAltitude'),
                    yaw=(xv('GimbalYawDegree') if a.yaw_source == 'gimbal' else xv('FlightYawDegree')),
                    focal=rat(e1[piexif.ExifIFD.FocalLength]), camera=e0[piexif.ImageIFD.Model].decode('ascii', 'ignore').strip('\x00 '),
                    datetime=e1.get(piexif.ExifIFD.DateTimeOriginal, b'').decode('ascii', 'ignore'))

    for i, fp in enumerate(files):
        fn = os.path.basename(fp)
        if fn in done: continue
        done.add(fn)
        if time.time() - t_start > a.budget:
            print('budget reached at', i); break
        t0 = time.time()
        m = read_meta(fp)
        row = dict(run=a.run, model=(os.path.basename(a.model) if a.model and a.model.lower() != 'sobel' else (a.model or 'grayscale')), mode=a.mode, route=a.route,
                   idx=i, file=fn, datetime=m['datetime'], camera=m['camera'], eligible=1, reason='',
                   lat_true=m['lat'], lon_true=m['lon'], gps_alt=m['gps_alt'], rel_alt=m['rel_alt'], yaw=m['yaw'])
        cam = CAMERAS.get(m['camera'])
        if cam is None:
            row.update(eligible=0, reason='unknown_camera'); wr.writerow(row); continue
        # --- altitude / DEM position ---
        if 'alt0' not in st:  # initialisation at the first frame of the route (GNSS available)
            st['alt0'] = m['gps_alt'] - m['rel_alt'] if np.isfinite(m['rel_alt']) else m['gps_alt']
            c, r = mp.ll_to_px(m['lat'], m['lon'])
            if a.init_offset_m:
                dpx = a.init_offset_m / MAP_GSD / math.sqrt(2); c += dpx; r -= dpx
            st['prev'] = [c, r]; st['pred'] = [c, r]; st['win'] = a.win; st['src'] = 'gps_init'; st['lost'] = 0
        if a.mode == 'track_causal':
            alt_used = st['alt0'] + m['rel_alt']
            lat_d, lon_d = mp.px_to_ll(st['pred'][0], st['pred'][1])
        else:
            alt_used = m['gps_alt']; lat_d, lon_d = m['lat'], m['lon']
        alt_used += a.alt_offset
        dem_v = dem.sample(lat_d, lon_d)
        if not np.isfinite(dem_v):
            row.update(eligible=0, reason='dem_out_of_range'); wr.writerow(row); continue
        h_agl = (alt_used - dem_v + a.dem_offset) * a.alt_scale
        if h_agl < 30:
            row.update(eligible=0, reason='h_agl_too_low', alt_used=alt_used, dem_at=dem_v, h_agl=h_agl); wr.writerow(row); continue
        gsd = cam['sensor_w'] * h_agl / (m['focal'] * cam['img_w'])  # m/px
        scale = gsd / MAP_GSD
        row.update(alt_used=alt_used, dem_at=dem_v, h_agl=round(h_agl, 2), scale=round(scale, 5))
        # --- search window centre ---
        if a.mode == 'oracle':
            wcx, wcy = mp.ll_to_px(m['lat'], m['lon']); wpx = a.oracle_win; src = 'gps'
        else:
            wcx, wcy = st['pred']; wpx = st['win']; src = st['src']
        if not (272 <= wcx < mp.w - 272 and 272 <= wcy < mp.h - 272):
            row.update(eligible=0, reason='window_centre_off_map', win_cx=wcx, win_cy=wcy, win_px=wpx); wr.writerow(row); continue
        # --- frame normalisation ---
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            row.update(eligible=0, reason='unreadable'); wr.writerow(row); continue
        if a.undistort and m['camera'] == 'L1D-20c':
            key = img.shape
            if key not in maps_undist:
                maps_undist[key] = cv2.initUndistortRectifyMap(K_L1D, D_L1D, None, K_L1D, (img.shape[1], img.shape[0]), cv2.CV_16SC2)
            img = cv2.remap(img, maps_undist[key][0], maps_undist[key][1], cv2.INTER_LINEAR)
        yaw = a.yaw_sign * m['yaw'] + a.yaw_offset
        rot = rotate_inscribed(img, yaw)  # cv2 rotates CCW for positive angle
        nh, nw = int(rot.shape[0] * scale), int(rot.shape[1] * scale)
        if min(nh, nw) < TILE:
            row.update(eligible=0, reason='frame_smaller_than_tile_after_scaling'); wr.writerow(row); continue
        norm = cv2.resize(rot, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        cx, cy = nw // 2, nh // 2
        # --- visual odometry prior (track modes): frame-to-frame translation by phase correlation ---
        vo_dx = vo_dy = 0.0; vo_resp = float('nan'); vo_src = 'none'
        if a.mode != 'oracle' and a.vo:
            v = min(a.vo_size, nh, nw)
            cur_full = norm[cy - v // 2: cy + v // 2, cx - v // 2: cx + v // 2]
            prev_full = vo_prev.get('img'); pidx = vo_prev.get('idx')
            if prev_full is not None and pidx == i - 1:
                vc = min(v, prev_full.shape[0], prev_full.shape[1])  # common central region
                def _c(img):
                    h, w = img.shape; return cv2.resize(img[h // 2 - vc // 2: h // 2 + vc // 2, w // 2 - vc // 2: w // 2 + vc // 2], (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
                (sx, sy), resp = cv2.phaseCorrelate(_c(prev_full), _c(cur_full), vo_win)
                vo_resp = float(resp)
                if resp >= a.vo_min_resp and math.hypot(sx, sy) < 100:
                    vo_dx, vo_dy = -sx * vc / 256.0, -sy * vc / 256.0; vo_src = 'vo'
                    st['vel'] = [vo_dx, vo_dy]
                elif st.get('vel'):
                    vo_dx, vo_dy = st['vel']; vo_src = 'coast'
            elif st.get('vel'):
                vo_dx, vo_dy = st['vel']; vo_src = 'coast'
            vo_prev.update(img=cur_full.copy(), idx=i)
            st['pred'] = [st['pred'][0] + vo_dx, st['pred'][1] + vo_dy]
            wcx, wcy = st['pred']
            c0, r0 = wcx - wpx / 2, wcy - wpx / 2
            if not (272 <= wcx < mp.w - 272 and 272 <= wcy < mp.h - 272):
                row.update(eligible=0, reason='window_centre_off_map', win_cx=wcx, win_cy=wcy, win_px=wpx); wr.writerow(row); continue
        row.update(vo_dx=round(vo_dx, 1), vo_dy=round(vo_dy, 1), vo_resp=round(vo_resp, 3) if np.isfinite(vo_resp) else '', vo_src=vo_src)
        d = min(200, min(cx, cy) - 272)
        crops = [norm[cy - 272 - d: cy + 272 - d, cx - 272 - d: cx + 272 - d],
                 norm[cy - 272: cy + 272, cx - 272: cx + 272],
                 norm[cy - 272 + d: cy + 272 + d, cx - 272 + d: cx + 272 + d]]
        if cae:
            tpls = [t[BORDER:TILE - BORDER, BORDER:TILE - BORDER] for t in cae.run(crops, equalize=bool(a.eq))]
        else:
            tpls = [t[BORDER:TILE - BORDER, BORDER:TILE - BORDER] for t in crops]
        # --- search ---
        c0, r0 = wcx - wpx / 2, wcy - wpx / 2
        win, c0, r0 = mp.read(c0, r0, wpx, wpx)
        boxes, peaks, scores, on_border = [], [], [], 0
        Hw, Ww = win.shape
        for j, t in enumerate(tpls):
            mres = match_in_window(win, t)
            if mres is None: boxes.append(None); peaks.append(None); scores.append(float('nan')); continue
            x, y, s = mres
            scores.append(s)
            if x <= 1 or y <= 1 or x >= Ww - TPL - 1 or y >= Hw - TPL - 1:
                on_border += 1; boxes.append(None); peaks.append(None); continue  # peak on the window border: unreliable
            boxes.append((c0 + x, r0 + y, TPL, TPL)); peaks.append((c0 + x, r0 + y))
        cons = consensus_loose(boxes) if a.consensus == 'loose' else consensus_strict(peaks, d, a.tol)
        row.update(win_cx=round(wcx, 1), win_cy=round(wcy, 1), win_px=wpx, win_center_src=src, disp_px=d, border=on_border,
                   score_ul=round(scores[0], 4), score_c=round(scores[1], 4), score_lr=round(scores[2], 4))
        reject = ''
        if cons is not None and a.mode != 'oracle' and a.gate > 0:
            ecx, ecy, n = cons
            px, py = st['pred']
            gpx = min(a.gate + a.gate_growth * st.get('lost', 0), wpx / 2.0)
            if vo_src == 'none': gpx = wpx / 2.0  # no motion information: only the window itself constrains the fix
            dist = math.hypot(ecx - px, ecy - py)
            row.update(gate_px=round(gpx), pred_dist_px=round(dist))
            if dist > gpx: reject = 'gate'
        if cons is None or reject:
            row.update(resolved=0, n_agree=0 if cons is None else cons[2], reason=('no_consensus' if cons is None else reject))
            if cons is not None:  # record the rejected candidate for the analysis
                lat_e, lon_e = mp.px_to_ll(cons[0], cons[1])
                row.update(est_col=round(cons[0], 1), est_row=round(cons[1], 1), lat_est=lat_e, lon_est=lon_e,
                           err_m=round(haversine(m['lat'], m['lon'], lat_e, lon_e), 2))
            if a.mode != 'oracle':
                st['lost'] = st.get('lost', 0) + 1
                st['win'] = min(a.win_max, a.win + a.grow * st['lost']); st['src'] = 'pred_grown'
        else:
            ecx, ecy, n = cons
            lat_e, lon_e = mp.px_to_ll(ecx, ecy)
            err = haversine(m['lat'], m['lon'], lat_e, lon_e)
            row.update(resolved=1, n_agree=n, est_col=round(ecx, 1), est_row=round(ecy, 1), lat_est=lat_e, lon_est=lon_e, err_m=round(err, 2))
            if a.mode != 'oracle':
                st['prev'] = [ecx, ecy]; st['pred'] = [ecx, ecy]; st['lost'] = 0
                st['win'] = a.win; st['src'] = 'pred'
        row['sec'] = round(time.time() - t0, 2)
        wr.writerow(row); fout.flush()
        json.dump(st, open(state_path, 'w'))
        print(f"[{i + 1}/{len(files)}] {fn} {'ERR=%.0f m' % row['err_m'] if row.get('err_m') is not None else 'UNRESOLVED'} "
              f"n={row.get('n_agree')} win={wpx} hagl={h_agl:.0f} scale={scale:.3f} {row['sec']}s", flush=True)
    fout.close()


if __name__ == '__main__':
    main()
