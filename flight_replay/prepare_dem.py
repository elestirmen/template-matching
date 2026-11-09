"""Make tiled copies of the (striped) DEM rasters so that single-pixel reads are cheap.
Usage: python prepare_dem.py SRC.tif OUT.tif
"""
import sys, os, rasterio
src, out = sys.argv[1], sys.argv[2]
if os.path.exists(out):
    print('exists:', out); sys.exit(0)
with rasterio.open(src) as ds:
    prof = ds.profile.copy()
    prof.update(tiled=True, blockxsize=256, blockysize=256, compress='LZW', BIGTIFF='YES')
    print('reading', src, ds.width, 'x', ds.height, ds.dtypes[0], 'tiled=', ds.is_tiled)
    data = ds.read(1)
    with rasterio.open(out, 'w', **prof) as dst:
        dst.write(data, 1)
print('wrote', out)
