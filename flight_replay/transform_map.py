"""Transform a georeferenced mosaic into a structural reference raster (CAE or Sobel baseline).
Tiling: 544x544 input tiles on a 512 px grid (32 px overlap); 16 px removed from every
output edge; stitched without blending. Resumable (rows already written are skipped).
Usage: python transform_map.py SRC.tif OUT.tif MODEL.onnx|sobel [--eq 1] [--rows a b] [--budget s]
"""
import sys, os, argparse, time, traceback, numpy as np, rasterio, cv2
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cae_infer import make_transform

ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('model')
ap.add_argument('--eq', type=int, default=1)
ap.add_argument('--rows', type=int, nargs=2, default=None)
ap.add_argument('--budget', type=float, default=1e9, help='seconds; stop before exceeding')
a = ap.parse_args()
T = 544; S = 512; B = 16
src = rasterio.open(a.src)
W, H = src.width, src.height
nbx, nby = -(-W // S), -(-H // S)
prog_path = a.out + '.rows_done'
done = set(int(l) for l in open(prog_path).read().split()) if os.path.exists(prog_path) else set()
if len(done) >= nby and os.path.exists(a.out):
    print('complete:', a.out); sys.exit(0)
if not os.path.exists(a.out):
    prof = src.profile.copy()
    prof.update(count=1, dtype='uint8', tiled=True, blockxsize=512, blockysize=512, compress=None, nodata=None, BIGTIFF='YES')
    with rasterio.open(a.out, 'w', **prof) as dst:
        pass
    done = set()
rows = range(*a.rows) if a.rows else range(nby)
try:
    cae = make_transform(a.model)
except Exception:
    with open(a.out + '.error.log', 'a') as f: f.write(traceback.format_exc())
    raise
print('transform:', a.model, getattr(cae, 'provider', ''), flush=True)
t0 = time.time()
try:
  with rasterio.open(a.out, 'r+') as dst:
    for by in rows:
        if by in done: continue
        if time.time() - t0 > a.budget: print('budget reached at row', by); break
        y0 = by * S - B
        ys, ye = y0, y0 + T
        pad_top = max(0, -ys); pad_bot = max(0, ye - H)
        rr = src.read(window=Window(0, max(ys, 0), W, min(ye, H) - max(ys, 0)))
        if rr.shape[0] >= 3:
            strip = cv2.cvtColor(np.transpose(rr[:3], (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        else:
            strip = rr[0]
        strip = np.pad(strip, ((pad_top, pad_bot), (B, T - S - B + max(0, nbx * S - W))), mode='reflect')
        tiles = [strip[:, bx * S: bx * S + T] for bx in range(nbx)]
        outs = []
        for i in range(0, len(tiles), 8):
            outs += cae.run(tiles[i:i + 8], equalize=bool(a.eq))
        row_out = np.hstack([o[B:T - B, B:T - B] for o in outs])[:, :W]
        hrow = min(S, H - by * S)
        dst.write(row_out[:hrow][None], window=Window(0, by * S, W, hrow))
        done.add(by)
        with open(prog_path, 'a') as f: f.write(f'{by}\n')
        print(f'row {by}/{nby} done ({time.time() - t0:.0f}s)', flush=True)
except Exception:
    with open(a.out + '.error.log', 'a') as f: f.write(traceback.format_exc())
    raise
print('rows done:', len(done), '/', nby)
