"""
Bu betik, anlık (drone) görüntülerini referans ortofoto/harita üzerinde konumlandırır.

Özet akış:
- EXIF'ten yaw (uçuş başı), GPS, odak uzaklığı ve irtifa bilgisi okunur.
- DEM (sayısal yükseklik modeli) ile farklı noktalardaki rakım farkı dikkate alınarak üç ölçekli şablon oluşturulur.
- Keras modeli ile bu şablonlardan özellik/olasılık haritaları üretilir.
- Template Matching ile ana haritada en iyi eşleşmeler bulunur; üç eşleşmenin kesişiminden konum çıkarılır.
- Hata metrikleri (RMSE/MAE/std) ve sınıflandırma istatistikleri (TP/FP/TN/FN, F-skor) hesaplanır ve dosyaya yazılır.
- İsteğe bağlı CUDA hızlandırması ile bazı adımlar GPU'da çalıştırılır.

Klasörler:
- `haritalar/`: Ana haritalar (DEM ve ortofoto)
- `model/`: Keras modelleri (544x544 girişli, kenar kırpmalı çıkış)
- `parcalar/`: Anlık görüntüler (EXIF içeren JPG/PNG)
"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
import cv2
from osgeo import gdal
#import exiftool
from tensorflow.keras.models import load_model
import pickle
import multiprocessing
import warnings
import math
import time
import rasterio as rio
import numpy as np
from math import cos, sqrt
import piexif
import csv
from PIL import Image
from PIL.ExifTags import TAGS
from affine import Affine
from pyproj import Transformer
import concurrent.futures
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Yardımcı fonksiyonlar grupları
# - CUDA kontrolü ve hızlandırılmış işlemler (resize / template matching)
# - EXIF/GPS okuma ve dönüşümler (WGS84 <-> UTM, piksel <-> koordinat)
# - Basit geometri ve metrikler (kesişim, RMSE/MAE/std, Haversine)
# - Görsel arayüz yardımcıları (HUD paneli, ölçek çubuğu, işaret çizimi)
# -----------------------------------------------------------------------------

# import rasterio as rio
# from rasterio.warp import transform 
# import matplotlib.pyplot as plt
# import pandas as pd
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img



def _cuda_available():
    try:
        return hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def cuda_resize_if_available(img, dsize, interpolation=cv2.INTER_NEAREST):
    if img is None:
        return img
    if _cuda_available() and img.ndim in (2, 3) and img.dtype == np.uint8 and hasattr(cv2.cuda, 'resize'):
        try:
            g = cv2.cuda_GpuMat()
            g.upload(img)
            g_out = cv2.cuda.resize(g, dsize, interpolation=interpolation)
            return g_out.download()
        except Exception:
            pass
    return cv2.resize(img, dsize, interpolation=interpolation)


def log_cuda_info_once():
    try:
        if getattr(log_cuda_info_once, "_logged", False):
            return
        devices = 0
        has_tm = False
        try:
            if hasattr(cv2, 'cuda'):
                devices = cv2.cuda.getCudaEnabledDeviceCount()
                has_tm = hasattr(cv2.cuda, 'createTemplateMatching')
        except Exception:
            pass
        try:
            build_has_cuda = 'CUDA: YES' in cv2.getBuildInformation()
        except Exception:
            build_has_cuda = False
        print(f"OpenCV CUDA devices: {devices}, has TM: {has_tm}, build CUDA: {build_has_cuda}")
        log_cuda_info_once._logged = True
    except Exception:
        pass

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)


def dosyaya_yaz_t(sonuclar,dogru_tahmin,yanlis_tahmin):    
    
    #model_name="sonuclar_"+model_name
    sonuclar_dosya = open("sonuclar.txt", "w")
    # sonuclar = np.vstack((sonuclar,dogru_tahmin, yanlis_tahmin)).T
    # print(sonuclar)
    
    df = pd.DataFrame(sonuclar, columns=['goruntu', 'sonuc', 'gercek_latitude', 'gercek_longitude', 'tahmini_latitude', 'tahmini_longitude'])
    
    # df.loc[len(df.index)] = ["","",str(dogru_tahmin)+" dogru", str(yanlis_tahmin)+" yanlis"] 

    sonuclar_dosya.write(df.to_string())
    sonuclar_dosya.close()
    
    df.to_csv("sonuclar.csv", index=False)
    
    


import pandas as pd

def dosyaya_yaz(sonuclar, dogru_tahmin, yanlis_tahmin):
    
    # Veri çerçevesini oluştur
    df = pd.DataFrame(sonuclar, columns=['goruntu', 'sonuc', 'gercek_latitude', 'gercek_longitude', 'tahmini_latitude', 'tahmini_longitude','ucus_yuksekligi'])
    
    # Eğer her bir hücre bir liste içeriyorsa, bu listelerin ilk elemanını al
    for column in df.columns:
        df[column] = df[column].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Metin dosyasına yaz
    with open("sonuclar.txt", "w") as sonuclar_dosya:
        sonuclar_dosya.write(df.to_string())
    
    # CSV dosyasına kaydet
    df.to_csv("sonuclar.csv", index=False)


    
#exif bilgisi okur    
def get_field (exif,field) :
  for (k,v) in exif.items():
     if TAGS.get(k) == field:
        return v
 
 #gos coordinatını decimal sisteme çevirir
def conversion(yon,coord):
    direction = {'N':1, 'S':-1, 'E': 1, 'W':-1}  
    
    return (int(coord[0])+int(coord[1])/60.0+float(coord[2])/3600.0) * direction[yon]



def piksel_bul(path, longitude, latitude):
    """
    Find the row and column of a geographic coordinate in a raster file.
    
    Parameters:
    path (str): The path to the raster file.
    longitude (float): The longitude of the geographic coordinate.
    latitude (float): The latitude of the geographic coordinate.
    
    Returns:
    tuple: A tuple containing the row and column indices.
    """
    # Open the raster file
    with rio.open(path) as image_data:
        # Get the CRS for the raster file
        to_crs = image_data.crs

        # Initialize the transformer with high precision
        transformer = Transformer.from_crs("EPSG:4326", to_crs, always_xy=True)
        
        # Transform the coordinates
        new_x, new_y = transformer.transform(longitude, latitude)
        
        # Get the row and column index
        row, col = image_data.index(new_x, new_y)
        
    return row, col
    



def piksel_bul_fast(image_data, transformer, longitude, latitude):
    """
    Fast pixel lookup using pre-opened rasterio dataset and pre-built transformer.
    Returns (row, col).
    """
    new_x, new_y = transformer.transform(longitude, latitude)
    row, col = image_data.index(new_x, new_y)
    return row, col

def quick_distance(Lat1, Long1, Lat2, Long2):
    x = Lat2 - Lat1
    y = (Long2 - Long1) * cos((Lat2 + Lat1)*0.00872664626)  
    return 87.11 * sqrt(x*x + y*y)    #return 111.319 * sqrt(x*x + y*y)
    
def quick_distance_utm(Lat1, Long1, Lat2, Long2):
    """
    Calculate the approximate distance between two points in UTM Zone 36.
    
    Parameters:
    - Lat1, Long1: Latitude and Longitude of the first point in decimal degrees.
    - Lat2, Long2: Latitude and Longitude of the second point in decimal degrees.
    
    Returns:
    - float: The approximate distance between the two points in meters.
    """
    
    # Create a Transformer object for WGS84 to UTM Zone 36 conversion
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)
    
    # Convert the coordinates to UTM Zone 36
    easting1, northing1 = transformer.transform(Long1, Lat1)
    easting2, northing2 = transformer.transform(Long2, Lat2)
    
    # Calculate the approximate distance using UTM coordinates
    x = easting2 - easting1
    y = northing2 - northing1
    
    return sqrt(x ** 2 + y ** 2)    
    
    
    
def latlon_to_utm(latitude, longitude, zone_number=None, hemisphere=None):
    """
    Convert Latitude and Longitude to UTM coordinates.
    
    Parameters:
    - latitude (float): The latitude in decimal degrees.
    - longitude (float): The longitude in decimal degrees.
    - zone_number (int, optional): UTM zone number. If None, it will be calculated based on longitude.
    - hemisphere (str, optional): 'N' for Northern Hemisphere, 'S' for Southern Hemisphere. If None, it will be calculated based on latitude.
    
    Returns:
    - tuple: A tuple containing UTM Easting, UTM Northing, Zone Number, and Hemisphere.
    """
    
    # Calculate the UTM zone number if not provided
    if zone_number is None:
        zone_number = int((longitude + 180) / 6) + 1
    
    # Determine the hemisphere if not provided
    if hemisphere is None:
        hemisphere = 'N' if latitude >= 0 else 'S'
    
    # Select EPSG code by hemisphere
    epsg = 32600 + zone_number if hemisphere.upper() == 'N' else 32700 + zone_number
    tfm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = tfm.transform(longitude, latitude)
    return easting, northing, zone_number, hemisphere


from math import sin, cos, sqrt, atan2, radians

# ----------------------------
# Helpers: EXIF, cropping, CRS
# ----------------------------

def _to_float_ratio(val):
    try:
        # PIL may return tuples like (num, den)
        if isinstance(val, tuple) and len(val) == 2:
            num, den = val
            den = den or 1
            return float(num) / float(den)
        return float(val)
    except Exception:
        return None

def parse_exif(image_path):
    """Parse EXIF safely; returns dict with keys:
    yaw, latitude, longitude, altitude, focal_length, model.
    Returns None on failure.
    """
    import re
    try:
        with Image.open(image_path) as im:
            exif = im._getexif() or {}
    except Exception:
        return None

    def get_field(exif_dict, field):
        for (k, v) in exif_dict.items():
            if TAGS.get(k) == field:
                return v
        return None

    yaw = 0.0
    mk = get_field(exif, 'MakerNote')
    if mk is not None:
        s = str(mk)
        m = re.search(r'FlightDegree[^0-9]*([0-9]+)', s)
        if m:
            try:
                yaw = int(m.group(1)) / 10.0
            except Exception:
                yaw = 0.0

    gps = get_field(exif, 'GPSInfo') or {}
    lat_ref = gps.get(1, 'N')
    lat_val = gps.get(2)
    lon_ref = gps.get(3, 'E')
    lon_val = gps.get(4)
    alt_val = gps.get(6)
    alt_ref = gps.get(5, 0)

    def conv(ref, coord):
        if coord is None:
            return None
        try:
            d = _to_float_ratio(coord[0])
            m = _to_float_ratio(coord[1])
            s = _to_float_ratio(coord[2])
            if None in (d, m, s):
                return None
            sign = 1 if ref in ('N', 'E') else -1
            return sign * (d + m/60.0 + s/3600.0)
        except Exception:
            return None

    latitude = conv(lat_ref, lat_val)
    longitude = conv(lon_ref, lon_val)

    altitude = _to_float_ratio(alt_val)
    if altitude is not None and alt_ref == 1:
        altitude = -altitude

    fl = get_field(exif, 'FocalLength')
    focal_length = _to_float_ratio(fl)
    model = get_field(exif, 'Model')

    return {
        'yaw': yaw,
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude,
        'focal_length': focal_length,
        'model': model,
    }

def make_rc_to_ll(dataset):
    """Create a pixel(row,col)->(lon,lat) converter for a rasterio dataset."""
    T1 = dataset.transform * Affine.translation(0.5, 0.5)
    to_wgs = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
    def rc_to_ll(row, col):
        x, y = (col, row) * T1
        lon, lat = to_wgs.transform(x, y)
        return lon, lat
    return rc_to_ll

def is_valid_slice(img, x1, y1, x2, y2):
    return x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]

def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Compute differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distance
    distance = R * c
    
    return distance



"""
UI helpers: panels, scale bar, minimal HUD
"""
def _draw_alpha_panel(img, x0, y0, x1, y1, color=(0, 0, 0), alpha=0.5):
    """Draw a filled rectangle with alpha blending onto img in-place."""
    x0 = max(0, min(int(x0), img.shape[1]-1))
    x1 = max(0, min(int(x1), img.shape[1]-1))
    y0 = max(0, min(int(y0), img.shape[0]-1))
    y1 = max(0, min(int(y1), img.shape[0]-1))
    if x1 <= x0 or y1 <= y0:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def draw_info_panel(img, lines, top_left=(25, 150), font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=6, thickness=20, text_color=(255, 255, 255),
                    bg_color=(0, 0, 0), alpha=0.5, padding=25, line_gap=None):
    """Draw a semi-transparent info panel with multiple text lines.

    lines: list of strings to show, one per line
    top_left: baseline of the first text line (x, y)
    """
    if not lines:
        return
    try:
        sizes = [cv2.getTextSize(str(s), font, font_scale, thickness)[0] for s in lines]
    except Exception:
        return
    max_w = max(w for (w, h) in sizes)
    max_h = max(h for (w, h) in sizes)
    if line_gap is None:
        line_gap = int(max_h * 1.6)

    x, y = top_left
    panel_w = max_w + 2 * padding
    panel_h = line_gap * len(lines) + padding
    panel_x0 = max(0, x - padding)
    panel_y0 = max(0, y - max_h - padding)
    panel_x1 = min(img.shape[1]-1, panel_x0 + panel_w)
    panel_y1 = min(img.shape[0]-1, panel_y0 + panel_h)
    _draw_alpha_panel(img, panel_x0, panel_y0, panel_x1, panel_y1, color=bg_color, alpha=alpha)

    for i, s in enumerate(lines):
        org = (int(x), int(y + i * line_gap))
        cv2.putText(img, str(s), org, font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_scale_bar(img, cpp_cm_per_px, scale_meters=100, margin=60, bar_height=35,
                   color=(255, 255, 255), text_color=(255, 255, 255),
                   font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=3, thickness=8):
    """Draw a metric scale bar in the bottom-right corner.

    cpp_cm_per_px: centimeters per pixel (float). If None/0, function does nothing.
    scale_meters: length of the scale bar in meters.
    """
    try:
        cpp = float(cpp_cm_per_px)
    except Exception:
        return
    if cpp <= 0:
        return
    bar_w = int((scale_meters * 100.0) / cpp)  # px
    if bar_w < 20:
        return
    h, w = img.shape[:2]
    x1 = max(0, w - margin - bar_w)
    y1 = max(0, h - margin - bar_height)
    x2 = min(w - 1, x1 + bar_w)
    y2 = min(h - 1, y1 + bar_height)

    # Background for readability
    _draw_alpha_panel(img, x1 - 25, y1 - 80, x2 + 25, y2 + 25, color=(0, 0, 0), alpha=0.5)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=max(2, thickness // 2))  # border

    label = f"{int(scale_meters)} m"
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    tx = x1 + (bar_w - tw) // 2
    ty = max(th + 10, y1 - 15)
    cv2.putText(img, label, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_plane_icon(img, center, heading_deg, size_px=180,
                    color=(255, 0, 255), outline=(0, 0, 0), outline_thickness=6):
    """Gerçek konumu bir uçak simgesi ile göstermek için basit bir ikon çizer.

    - center: (x, y) piksel koordinatı (örn. (knm[1], knm[0]))
    - heading_deg: yaw/heading (derece). 0=Kuzey, pozitif saat yönü varsayımı ile -yaw uygulanır.
    - size_px: ikonun uzunluğu (burun-kuyruk arası) piksel cinsinden.
    """
    try:
        cx, cy = int(center[0]), int(center[1])
        h = float(size_px)
        w = h * 0.7

        # Basit uçak silueti (burun yukarı bakacak şekilde tanımlı)
        pts = np.array([
            [0.0, -0.5 * h],      # burun
            [0.5 * w, -0.15 * h], # sağ kanat ucu
            [0.25 * w, 0.50 * h], # sağ kuyruk
            [0.0, 0.33 * h],      # gövde alt
            [-0.25 * w, 0.50 * h],# sol kuyruk
            [-0.5 * w, -0.15 * h] # sol kanat ucu
        ], dtype=np.float32)

        # İkonu heading'e göre döndür (OpenCV pozitif=CCW, yaw pozitif= saat yönü varsayımıyla -yaw)
        ang = float(heading_deg)
        rad = np.deg2rad(ang)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts_rot = (pts @ R.T)
        pts_rot[:, 0] += cx
        pts_rot[:, 1] += cy
        pts_i = pts_rot.astype(np.int32)

        # Doldur + kenarlık
        cv2.fillPoly(img, [pts_i], color)
        if outline_thickness > 0:
            cv2.polylines(img, [pts_i], isClosed=True, color=outline, thickness=outline_thickness, lineType=cv2.LINE_AA)
    except Exception:
        # Hata durumunda bir yedek işaret bırak (küçük daire)
        try:
            cv2.circle(img, (cx, cy), 10, color, 2)
        except Exception:
            pass




def draw_plane_icon_v2(img, center, heading_deg, size_px=200,
                       color=(255, 0, 255), outline=(0, 0, 0), outline_thickness=8):
    """Gerçek konumu bir uçak simgesi ile göstermek için daha gerçekçi bir ikon çizer.

    - center: (x, y) piksel koordinatı (örn. (knm[1], knm[0]))
    - heading_deg: yaw/heading (derece). 0=Kuzey, pozitif saat yönü varsayımıyla -yaw uygulanır.
    - size_px: ikonun toplam uzunluğu (burun-kuyruk).

    PNG desteği: Eğer çalışma klasöründe `plane_icon.png`/`plane.png` (veya `assets/` altında) varsa,
    otomatik olarak bu PNG döndürülerek alfa ile bindirilir. Yoksa vektörel bir siluet çizilir.
    """
    try:
        cx, cy = int(center[0]), int(center[1])

        # 1) Önce varsa PNG ikonunu kullan
        icon = None
        try:
            candidates = [
                os.getenv('PLANE_ICON_PNG'),
                os.path.join(dirname, 'plane_icon.png'),
                os.path.join(dirname, 'plane.png'),
                os.path.join(dirname, 'assets', 'plane_icon.png'),
                os.path.join(dirname, 'assets', 'plane.png'),
            ]
            for p in candidates:
                if p and os.path.exists(p):
                    icon = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    if icon is not None and icon.size > 0:
                        break
        except Exception:
            icon = None

        if icon is not None and icon.ndim in (2, 3, 4):
            # Döndür ve merkezde bindir
            # 0 derece = Kuzey olacak şekilde hizala (PNG başı genelde sağ/East varsayıldığında +90)
            ang = float(heading_deg) - 90.0
            ih, iw = icon.shape[:2]
            M = cv2.getRotationMatrix2D((iw/2.0, ih/2.0), ang, 1.0)
            abs_cos = abs(M[0,0]); abs_sin = abs(M[0,1])
            nW = int(ih * abs_sin + iw * abs_cos)
            nH = int(ih * abs_cos + iw * abs_sin)
            M[0,2] += (nW/2.0) - iw/2.0
            M[1,2] += (nH/2.0) - ih/2.0
            rot = cv2.warpAffine(icon, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

            # Ölçekle (en büyük boyutu size_px olacak şekilde)
            scale = float(size_px) / max(nW, nH)
            if scale != 1.0:
                rot = cv2.resize(rot, (max(1,int(nW*scale)), max(1,int(nH*scale))), interpolation=cv2.INTER_AREA)
            rh, rw = rot.shape[:2]

            # ROI belirle ve alfa ile bindir
            x1 = cx - rw//2; y1 = cy - rh//2
            x2 = x1 + rw;   y2 = y1 + rh
            H, W = img.shape[:2]
            rx1 = max(0, x1); ry1 = max(0, y1)
            rx2 = min(W, x2); ry2 = min(H, y2)
            if rx1 < rx2 and ry1 < ry2:
                sx1 = rx1 - x1; sy1 = ry1 - y1; sx2 = sx1 + (rx2 - rx1); sy2 = sy1 + (ry2 - ry1)
                overlay = rot[sy1:sy2, sx1:sx2]
                base = img[ry1:ry2, rx1:rx2]
                if overlay.shape[2] == 4:
                    alpha = overlay[:,:,3:4].astype(np.float32) / 255.0
                    fg = overlay[:,:,:3].astype(np.float32)
                    bg = base.astype(np.float32)
                    comp = alpha * fg + (1.0 - alpha) * bg
                    img[ry1:ry2, rx1:rx2] = np.clip(comp, 0, 255).astype(np.uint8)
                else:
                    img[ry1:ry2, rx1:rx2] = overlay
            return

        # 2) PNG yoksa vektörel siluet çiz
        h = float(size_px)
        w = h * 0.7
        pts = np.array([
            [0.00*w, -0.60*h],   # burun
            [0.12*w, -0.45*h],   # gövde sağ (ileri)
            [0.18*w, -0.15*h],   # gövde sağ (kanat öncesi)
            [0.60*w,  0.02*h],   # sağ kanat ucu
            [0.20*w,  0.08*h],   # sağ kanat arkası
            [0.18*w,  0.42*h],   # gövde sağ (arka)
            [0.35*w,  0.52*h],   # sağ yatay stabilize ucu
            [0.12*w,  0.55*h],   # kuyruk sağ
            [0.00*w,  0.60*h],   # kuyruk orta
            [-0.12*w, 0.55*h],   # kuyruk sol
            [-0.35*w, 0.52*h],   # sol yatay stabilize ucu
            [-0.18*w, 0.42*h],   # gövde sol (arka)
            [-0.20*w, 0.08*h],   # sol kanat arkası
            [-0.60*w, 0.02*h],   # sol kanat ucu
            [-0.18*w,-0.15*h],   # gövde sol (kanat öncesi)
            [-0.12*w,-0.45*h],   # gövde sol (ileri)
        ], dtype=np.float32)

        cockpit = np.array([
            [0.00*w, -0.52*h],
            [0.09*w, -0.40*h],
            [-0.09*w, -0.40*h],
        ], dtype=np.float32)

        # 0 derece = Kuzey (yukarı) olacak şekilde hizala
        ang = float(heading_deg)
        rad = np.deg2rad(ang)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)

        pts_rot = (pts @ R.T);
        cockpit_rot = (cockpit @ R.T)
        pts_rot[:, 0] += cx;  pts_rot[:, 1] += cy
        cockpit_rot[:, 0] += cx; cockpit_rot[:, 1] += cy
        pts_i = pts_rot.astype(np.int32)
        cp_i = cockpit_rot.astype(np.int32)

        cv2.fillPoly(img, [pts_i], color)
        cv2.fillPoly(img, [cp_i], (0, 0, 0))
        if outline_thickness > 0:
            cv2.polylines(img, [pts_i], isClosed=True, color=outline, thickness=outline_thickness, lineType=cv2.LINE_AA)
    except Exception:
        try:
            cv2.circle(img, (cx, cy), 10, color, 2)
        except Exception:
            pass


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform (prefer CUDA if available)
    try:
        _gpu = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2.cuda, 'warpAffine')
    except Exception:
        _gpu = False

    if _gpu and image.ndim == 2 and image.dtype == np.uint8:
        try:
            g_img = cv2.cuda_GpuMat()
            g_img.upload(image)
            g_res = cv2.cuda.warpAffine(g_img, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
            result = g_res.download()
            return result
        except Exception:
            pass

    result = cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """
    angle = math.radians(angle)
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)

#%%

#simulasyon olarak çalışması için true olarak ayarlayın, Benchmark için false olarak ayarlayın
# -----------------------------------------------------------------------------
# Global ayarlar ve sabitler
# - benchmark: Görsel ve adım adım akış yerine daha yalın/performans denemesi
# - PATCH_SIZE/PRED_BORDER: Modelin beklediği giriş ve çıktı kenar kırpma miktarı
# - USE_PYRAMID/COARSE_SCALE/ROI_PAD_FACTOR: Template Matching'i hızlandırma parametreleri
# -----------------------------------------------------------------------------
benchmark=False

# Global debug and sizing constants
DEBUG = False
PATCH_SIZE = 544
PATCH_HALF = PATCH_SIZE // 2
PRED_BORDER = 16
# CPU optimization flags
USE_PYRAMID = True
COARSE_SCALE = 0.5
ROI_PAD_FACTOR = 2.0

if benchmark==True:
    cerceve_boyutu_deger=5000
else:
    cerceve_boyutu_deger=2048



#%%

from math import radians, sin, cos, sqrt, atan2

def calculate_coordinates(latitude, longitude, d_lat, d_long):
    R = 6378137  # Yarıçapı metre cinsinden olan WGS-84 elipsoiti
    
    new_latitude = latitude + (d_lat / R) * (180 / 3.14159265358979323846)
    new_longitude = longitude + (d_long / (R * cos(3.14159265358979323846 * latitude / 180))) * (180 / 3.14159265358979323846)
    
    return new_latitude, new_longitude

def find_corner_coordinates(center_latitude, center_longitude, pixel_distance, GSD):
    # GSD (Ground Sample Distance): Metre cinsinden piksel başına düşen gerçek dünya uzunluğu
    # pixel_distance: Kaç piksel uzaklıkta yeni bir nokta oluşturulacağı
    
    distance = pixel_distance * GSD  # Metre cinsinden toplam mesafe
    
    # Sol üst köşe koordinatları
    new_latitude1, new_longitude1 = calculate_coordinates(center_latitude, center_longitude, -distance, -distance)
    
    # Sağ alt köşe koordinatları
    new_latitude2, new_longitude2 = calculate_coordinates(center_latitude, center_longitude, distance, distance)
    
    return (new_latitude1, new_longitude1), (new_latitude2, new_longitude2)



def match(img,template):
    # Prefer CUDA if available and enabled, otherwise fall back to CPU
    method = cv2.TM_CCOEFF_NORMED

    # Lazy-init CUDA flags on first call
    global _CUDA_TM_INITIALIZED, _CUDA_TM_AVAILABLE, _CUDA_TM_DISABLED
    try:
        _CUDA_TM_INITIALIZED
    except NameError:
        _CUDA_TM_INITIALIZED = False
        _CUDA_TM_AVAILABLE = False
        _CUDA_TM_DISABLED = False

    if not _CUDA_TM_INITIALIZED:
        try:
            _CUDA_TM_AVAILABLE = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2.cuda, 'createTemplateMatching')
        except Exception:
            _CUDA_TM_AVAILABLE = False
        _CUDA_TM_INITIALIZED = True

    if _CUDA_TM_AVAILABLE and not _CUDA_TM_DISABLED and img.dtype == np.uint8 and template.dtype == np.uint8 and img.ndim == 2 and template.ndim == 2:
        try:
            # Upload to GPU
            g_img = cv2.cuda_GpuMat()
            g_tmpl = cv2.cuda_GpuMat()
            g_img.upload(img)
            g_tmpl.upload(template)

            # Determine src type
            src_type = cv2.CV_8UC1 if hasattr(cv2, 'CV_8UC1') else cv2.CV_8U
            tm = cv2.cuda.createTemplateMatching(src_type, method)
            g_res = tm.match(g_img, g_tmpl)
            res = g_res.download()
            return res
        except Exception as e:
            print("CUDA template matching kullanılamadı, CPU'ya düşülüyor:", e)
            _CUDA_TM_DISABLED = True

    # CPU fallback
    res = cv2.matchTemplate(img, template, method, None)
    return res

def _init_cuda_tm_state():
    global _CUDA_TM_INITIALIZED, _CUDA_TM_AVAILABLE, _CUDA_TM_DISABLED
    try:
        _CUDA_TM_INITIALIZED
    except NameError:
        _CUDA_TM_INITIALIZED = False
        _CUDA_TM_AVAILABLE = False
        _CUDA_TM_DISABLED = False
    if not _CUDA_TM_INITIALIZED:
        try:
            _CUDA_TM_AVAILABLE = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2.cuda, 'createTemplateMatching')
        except Exception:
            _CUDA_TM_AVAILABLE = False
        _CUDA_TM_INITIALIZED = True

def match_three(img, templates):
    """Run template matching for three templates.
    If CUDA is available, upload img once and run three matches on GPU.
    Returns (res1, res2, res3).
    """
    method = cv2.TM_CCOEFF_NORMED
    _init_cuda_tm_state()

    # CPU fallback path
    if not _CUDA_TM_AVAILABLE or any(t.ndim != 2 or t.dtype != np.uint8 for t in templates) or img.ndim != 2 or img.dtype != np.uint8:
        try:
            if not getattr(match_three, "_logged_backend", False):
                print("TemplateMatching backend: CPU")
                match_three._logged_backend = True
        except Exception:
            pass

        img_c = np.ascontiguousarray(img)
        tmps_c = [np.ascontiguousarray(t) for t in templates]

        def _match_direct(a, b):
            return cv2.matchTemplate(a, b, method, None)

        def _match_pyramid(a, b):
            # Full result shape
            H, W = a.shape[:2]
            h, w = b.shape[:2]
            resH, resW = (H - h + 1, W - w + 1)
            if resH <= 0 or resW <= 0:
                return np.empty((0, 0), dtype=np.float32)

            # Coarse downscale
            s = COARSE_SCALE
            a_small = cv2.resize(a, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
            b_small = cv2.resize(b, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            res_small = cv2.matchTemplate(a_small, b_small, method, None)
            _, _, _, max_loc_small = cv2.minMaxLoc(res_small)

            # Map back to full-res match coordinates
            cx = int(max_loc_small[0] / s)
            cy = int(max_loc_small[1] / s)

            # ROI around coarse location in res-space
            pad = int(max(w, h) * ROI_PAD_FACTOR)
            x1 = max(0, cx - pad)
            y1 = max(0, cy - pad)
            x2 = min(resW - 1, cx + pad)
            y2 = min(resH - 1, cy + pad)

            # Corresponding image region for refined match
            img_x1 = x1
            img_y1 = y1
            img_x2 = x2 + w - 1
            img_y2 = y2 + h - 1
            img_x2 = min(img_x2, W - 1)
            img_y2 = min(img_y2, H - 1)

            # Ensure valid region
            if img_x2 - img_x1 + 1 < w or img_y2 - img_y1 + 1 < h:
                # Fallback to direct if ROI is degenerate
                return cv2.matchTemplate(a, b, method, None)

            a_roi = a[img_y1:img_y2 + 1, img_x1:img_x2 + 1]
            res_roi = cv2.matchTemplate(a_roi, b, method, None)

            # Paste into full-sized result with very low value outside
            res_full = np.empty((resH, resW), dtype=res_roi.dtype)
            res_full.fill(-1e9)
            res_full[y1:y2 + 1, x1:x2 + 1] = res_roi
            return res_full

        if USE_PYRAMID:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futs = [ex.submit(_match_pyramid, img_c, t) for t in tmps_c]
                return tuple(f.result() for f in futs)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futs = [ex.submit(_match_direct, img_c, t) for t in tmps_c]
                return tuple(f.result() for f in futs)

    # GPU path
    try:
        g_img = cv2.cuda_GpuMat()
        g_img.upload(img)
        src_type = cv2.CV_8U
        # Cache TemplateMatching object
        if not hasattr(match_three, "_tm") or match_three._tm is None:
            match_three._tm = cv2.cuda.createTemplateMatching(src_type, method)
        tm = match_three._tm

        results = []
        for t in templates:
            g_tmpl = cv2.cuda_GpuMat()
            g_tmpl.upload(t)
            g_res = tm.match(g_img, g_tmpl)
            results.append(g_res.download())
        # Log once which backend used
        try:
            if not getattr(match_three, "_logged_backend", False):
                print("TemplateMatching backend: CUDA")
                match_three._logged_backend = True
        except Exception:
            pass
        return tuple(results)
    except Exception as e:
        print("CUDA TM kullanılmadı (hata), CPU'ya düşülüyor:", e)
        return tuple(cv2.matchTemplate(img, t, method, None) for t in templates)
    


dirname = os.path.dirname(os.path.abspath(__file__))


# RMSE hesaplama fonksiyonu
def rmse(errors):
    squared_errors = errors ** 2                     # hataların karesini al
    mean_squared_errors = squared_errors.mean()     # karelerin ortalamasını al
    rmse_val = np.sqrt(mean_squared_errors)          # Ortalamanın karekökünü al
    return rmse_val


# MAE hesaplama fonksiyonu
def mae(errors):
    absolute_errors = np.abs(errors)                # hataların mutlak değerini al
    mean_absolute_errors = absolute_errors.mean()   # mutlak hataların ortalamasını al
    return mean_absolute_errors

def standart_sapma(data):
    mean = np.mean(data)                     # Ortalamayı hesapla
    squared_diff = (data - mean) ** 2        # Ortalama ile farkların karesini al
    mean_squared_diff = np.mean(squared_diff)  # Kare farklarının ortalamasını al
    std_dev = np.sqrt(mean_squared_diff)     # Ortalamanın karekökünü al
    return std_dev





#%%





if __name__ == '__main__': 
    # Log CUDA environment once
    try:
        log_cuda_info_once()
    except Exception:
        pass
    
    #haritalar klasöründeki ilk görüntüde DEM verileri vardır. ikinci görüntü ise normal rgb görüntüdür.
    # 1) Yol/klasör hazırlığı: haritalar (ana harita), model (Keras), parcalar (anlık görüntüler)
    harita_yol=dirname+'/haritalar/'
    harita_yol_list=os.listdir(harita_yol)
    model_yol=dirname+'/model/'
    model_list=os.listdir(model_yol)
    #ana_harita_elevation = "urgup_genis_elevations.tif"
    #ana_harita_elevation="urgup_gmap_30_cm_elevations_560.tif"
    #ana_harita_elevation = "ana_harita_urgup_30_cm_elevation_544.tif"
    #ana_harita_elevation="ana_harita_karlik_30_cm_bingmap_elevations_576.tif"
    #ana_harita_elevation = "urgup_genel_genis_kendi_uretimim_elevation.tif"
    #ana_harita_elevation = "urgup_genis_karma_srtm_kendi_uretimim_elevation_mean.tif"
    ana_harita_elevation = "ana_harita_urgup_30_cm_utm_elevation.tif"
    #ana_harita_elevation = "karlik_30_cm_bingmap_utm_elevation.tif"
    
    
    
    
    
    # haritadaki piksellerin gps koordinatları bulunur ve koordinatlar olarak ayrı bri dosya olarak diske kaydedilir. bir kez çalıştırılması yeterlidir
    ###############################################################################
    #%%
    
    import rasterio
    from affine import Affine
 
    #fname = 'urgup_gmap_georef.tif'
    fname = harita_yol+harita_yol_list[0]
    # Read raster (metadata only)
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        T1 = T0 * Affine.translation(0.5, 0.5)
        to_wgs = Transformer.from_crs(r.crs, "EPSG:4326", always_xy=True)
    
    def koordinat_bul(row,col):
        # Convert pixel (row,col) to lon/lat using affine + CRS
        x, y = (col, row) * T1
        lon, lat = to_wgs.transform(x, y)
        return (np.array([lon]), np.array([lat]))
    
    
    #%%
    
    # pickle_in = open("koordinatlar.pickle","rb")
    # koordinatlar = pickle.load(pickle_in)
    
    
    # print(koordinatlar[0][10][10])
    # print(koordinatlar[1][10][10])
    ###############################################################################
    
    #DEM verileri aktarılır
    
    # 3) DEM rasterını (elevation) aç
    filename = ana_harita_elevation
            
    dataset = gdal.Open(filename)
    
    gt = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)  #5. bant elevation bandı
    
    DEM_array = band.ReadAsArray()
    # RasterIO dataset and transformer for DEM pixel lookup
    dem_ds = rio.open(filename)
    ll_to_dem = Transformer.from_crs("EPSG:4326", dem_ds.crs, always_xy=True)
    
    ###############################################################################
    
    #%%
    
    
    # 4) Başlangıç durumları ve toplayıcılar
    cerceve_boyutu=cerceve_boyutu_deger
    sonuclar = []
    
    konum=(0,0)
    konum_once=(0,0)
    kare=()  
    
    
    
    
    for k in range(len(harita_yol_list)):
        yanlis_pozitif=0
        dogru_pozitif=0
        dogru_negatif=0
        yanlis_negatif=0
        uzaklik_hatalari = []
        
        model_yolu=model_yol+model_list[k]                
        model = load_model(model_yolu)        
        
        
          
        dogru_tahmin=0
        yanlis_tahmin=0
        ana_harita="haritalar/"+harita_yol_list[k]
        # Referans haritayı gri-ton olarak oku (Template Matching için daha uygundur)
          
        t_img = cv2.imread(ana_harita,0)  #haritalar klasöründeki ikinci görüntüyü okur
        print(t_img.shape)

        # Open main map once per loop and reuse for pixel lookups
        map_ds = rio.open(ana_harita)
        ll_to_map = Transformer.from_crs("EPSG:4326", map_ds.crs, always_xy=True)
        rc_to_ll = make_rc_to_ll(map_ds)
          
        kenarx=int(t_img.shape[0]/512)
        
        #parcalar klasöründeki anlık görüntüleri getirir
        # 6) parcalar klasöründeki anlık görüntüleri getirir
        anlik_yol = os.getenv("ANLIK_YOL", os.path.join(dirname, 'parcalar'))
        
        #anlik_yol="parcalar/"
        
        anlik_yol_list=os.listdir(anlik_yol)
        
        #anlik_goruntu=anlik_yol+anlik_yol_list[0]
        
        # Dosyaları zamana göre sırala (akış sırasını korumak için pratik)
        anlik_yol_list = sorted( anlik_yol_list,
                                key = lambda x: os.path.getmtime(os.path.join(anlik_yol, x))  # tarihe göre klasördeki dosyaları sıralar
                                )
        # Liste elemanlarını tam yola çevir (örn. 'DJI_0001.JPG' -> '.../parcalar/DJI_0001.JPG')
        try:
            if anlik_yol_list and not os.path.isabs(anlik_yol_list[0]):
                anlik_yol_list = [os.path.join(anlik_yol, x) for x in anlik_yol_list]
        except Exception:
            pass
        uzaklik=0
        fark=100
        irtifa_dizisi=[]
        # 7) Her anlık görüntü için döngü
        for i in range(len(anlik_yol_list)):
            
            yanlis_pozitif_kontrol = 0            
            dogru_pozitif_kontrol = 0
            dogru_negatif_kontrol  = 0
            yanlis_negatif_kontrol = 0
            
            baslangic_zamani = time.time()

            
            konum_once=konum
            
            
            img = t_img
            
            print(img.shape)
            #anlik_goruntu = "parcalar/"+anlik_yol_list[i]  #klasördeki ilk görüntüyü getir
            anlik_goruntu = anlik_yol+anlik_yol_list[i]  #klasördeki ilk görüntüyü getir
            
            # Eğer liste elemanı zaten tam yol ise doğrudan kullan
            try:
                if os.path.isabs(anlik_yol_list[i]):
                    anlik_goruntu = anlik_yol_list[i]
            except Exception:
                pass

            # Yol düzeltme: liste elemanı + klasör ile doğru tam yolu dene
            try:
                _cand = os.path.join(anlik_yol, os.path.basename(anlik_yol_list[i]))
                if (not os.path.exists(anlik_goruntu)) and os.path.exists(_cand):
                    anlik_goruntu = _cand
            except Exception:
                pass
        
            #exif bilgileri okunur
            #####################################################
            # Güvenli yol birleştirme
            if not os.path.isabs(anlik_goruntu):
                anlik_goruntu = os.path.join(anlik_yol, os.path.basename(anlik_goruntu))
            if not os.path.exists(anlik_goruntu):
                print("dosya bulunamadı:", anlik_goruntu)
                continue
            anlik_img=Image.open(anlik_goruntu)
            exif = anlik_img._getexif()
            
            
            
            text = str(get_field(exif,'MakerNote'))
            
            x = int(text.find("FlightDegree"))          
            
            xx=text[x+20:x+25]
            
            #uçuş yönü bulunurken virgüle kadarki kısmı alır ve saaıya çevirir
            virgulsirasi=xx.find(",")
            if(virgulsirasi>0):
                yaw=float(xx[0:virgulsirasi])/10
            else:
                yaw=float(xx)/10
            
            
            GPSInfo =get_field(exif,'GPSInfo')
            
            altitude=float(GPSInfo[6])
            FocalLength = float(get_field(exif,'FocalLength'))
            gps_latitude_yon = GPSInfo[1]
            gps_latitude = GPSInfo[2]
            
            gps_longitude_yon = GPSInfo[3]
            gps_longitude = GPSInfo[4]
            
            gps_latitude,gps_longitude= (conversion(gps_latitude_yon,gps_latitude),conversion(gps_longitude_yon,gps_longitude))
            ######################################################
            kamera_model =get_field(exif,'Model')
           
            
           
            # Use pre-opened map and transformer for fast lookups
            # Harita üzerinde EXIF koordinatına karşılık gelen pikseli bul
            knm = piksel_bul_fast(map_ds, ll_to_map, gps_longitude, gps_latitude)
            

            
            
                       
                
            # İlk karede EXIF konumuna yakın çevrede, sonraki karelerde bir önceki tahmine yakın çevrede ara
            if benchmark==False:
                
                if i==0:
                    sol=-int(cerceve_boyutu/2)+knm[0]
                    sag=+int(cerceve_boyutu/2)+knm[0]
                    ust=-int(cerceve_boyutu/2)+knm[1]
                    alt=+int(cerceve_boyutu/2)+knm[1]
                    
                    if sol<0:
                        sol=0
                    if sag<0:
                        sag= 0
                    if ust<0:
                        ust= 0
                    if alt<0:
                        alt= 0
                    cerceve=img[sol:sag,ust:alt]                
                    konum=(knm[1],knm[0])
                else:
                    sol=-int(cerceve_boyutu/2)+konum[1]
                    sag=+int(cerceve_boyutu/2)+konum[1]
                    ust=-int(cerceve_boyutu/2)+konum[0]
                    alt=+int(cerceve_boyutu/2)+konum[0]
                    
                    if sol<0:
                        sol=0
                    if sag<0:
                        sag= 0
                    if ust<0:
                        ust= 0
                    if alt<0:
                        alt= 0
                    cerceve=img[sol:sag,ust:alt]
                
                    
            else:
                
                 cerceve_boyutu=cerceve_boyutu_deger
                 sol=-int(cerceve_boyutu/2)+knm[0]
                 sag=+int(cerceve_boyutu/2)+knm[0]
                 ust=-int(cerceve_boyutu/2)+knm[1]
                 alt=+int(cerceve_boyutu/2)+knm[1]
                 
                 if sol<0:
                     sol=0
                 if sag<0:
                     sag= 0
                 if ust<0:
                     ust= 0
                 if alt<0:
                     alt= 0
                 konum=(knm[1],knm[0])
                     
                 cerceve=img[sol:sag,ust:alt]
            
            
            
            
            # if cerceve.shape[0]==0 or cerceve.shape[1]==0:
            #     print("cerceve alan dışına çıktı")
            #     continue
            
            
            
            if knm[0]<272 or knm[0]>img.shape[0]-272:
                print("dışarıda")
                continue
            elif knm[1]<272 or knm[1]>img.shape[1]-272:
                print("dışarıda")
                continue
            
            
            
            sol_ust, sag_alt = find_corner_coordinates(gps_latitude, gps_longitude, 100, 0.30)            
            
            #anlık görüntünün ana haritada karşılık geldiği rakım değeri bulunur
            
            dem_konum = piksel_bul_fast(dem_ds, ll_to_dem, gps_longitude, gps_latitude)
            
            dem_konum_sol_ust = piksel_bul_fast(dem_ds, ll_to_dem, sol_ust[1], sol_ust[0])
            dem_konum_sag_alt = piksel_bul_fast(dem_ds, ll_to_dem, sag_alt[1], sag_alt[0])

            
            rakim=DEM_array[dem_konum[0],dem_konum[1]]
            rakim_sol_ust=DEM_array[dem_konum_sol_ust[0],dem_konum_sol_ust[1]]
            rakim_sag_alt=DEM_array[dem_konum_sag_alt[0],dem_konum_sag_alt[1]]
            
            
            
            
            print(rakim)
            
            
            
            """
            try:
                rakim=DEM_array[knm[1],knm[0]]
               
            except:
                print("dışarıda")
                continue
            """
            rakim_duzeltme=26
            if kamera_model=="L1D-20c":   
                #spatial çözünürlük elde etme
                #camera_sensor_genislik=15.9 #mavic2pro için 13.2  milimetre sensör genişliği
                camera_sensor_genislik = 13.2
                camera_focal_lenght = FocalLength #mavic2pro için 10.26 milimetre
                ucus_yuksekligi = altitude - rakim   + rakim_duzeltme    #+33 #metre olarak yerden x"x""uçuş yüksekliği  35 dem dosyasındaki hatadan dolayı
                goruntu_piksel_genisligi = 5472 #5472 #pipksel olarak resmin genişliği
                goruntu_piksel_yuksekligi = 3648 # 3648 #pipksel olarak resmin genişliği
                mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
                goruntunun_gercek_uzunlugu = (mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak               
                
                olcek_scale_test=(mekansal_cozunurluk/29.85)    
            
            
            elif kamera_model=="FC2204":   
                #spatial çözünürlük elde etme
                camera_sensor_genislik=6.17 #mavic2zoom için 6.17  milimetre sensör genişliği
                camera_focal_lenght= FocalLength  #mavic2zoom için 4 milimetre
                ucus_yuksekligi=altitude - rakim + rakim_duzeltme
                goruntu_piksel_genisligi =4000 # 4000 #pipksel olarak resmin genişliği
                goruntu_piksel_yuksekligi =3000 # 3000 #pipksel olarak resmin genişliği
                mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
                goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak             
                olcek_scale_test=(mekansal_cozunurluk/29.85)
                
                
            print("kamera model= ",kamera_model)
            print("focal lenght = ",FocalLength)
            print("altitude = : ",altitude)
            print("rakım =: ",rakim)
            print("ucus_yuksekligi = :",ucus_yuksekligi)
            print("yaw = :",yaw)
            
            
    
            
            #################################################################################################
            
            # 7.1) Anlık görüntüyü oku ve yaw/ölçek ile döndürmeye hazırla
            # Reading the image
            image = cv2.imread(anlik_goruntu,0)
            
            # dim=(1000,750)
            
            # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            
            # dividing height and width by 2 to get the center of the image
            height, width = image.shape[:2]
            # #get the center coordinates of the image to create the 2D rotation matrix
            #center = (int(width/2), int(height/2))
            
            # #using cv2.getRotationMatrix2D() to get the rotation matrix
            # #scale parametresi ile görüntünün spartial çözünürlüğü 60 cm'ye ayarlanır
            # #angle ile görüntünün yav değerinin tam tersine rotate edilir ve görüntü kuzeye döndürülür.
            # rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=(-1*yaw), scale=olcek_scale)
            
            
            # #rotate the image using cv2.warpAffine
            # rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderValue=(255,255,255))
            
            
            angle=-yaw
            rimage = rotate_image(image, angle)
            
            #cv2.imwrite("rotate_edilmis.jpg", rimage)
           
            
            t=largest_rotated_rect(width,height, angle)
            
            #cv2.imwrite("en_buyuk_ic_dortgen.jpg", t)
       
                
            #cv2.imshow("rotated",t)
            
            
            cr_image = crop_around_center(rimage,int(t[0]), int(t[1]))
            
            #cv2.imwrite("crop_edilmis.jpg", cr_image)
            
            
            #cv2.imshow("rotated",cr_image)
            
            height,width= (cr_image.shape[0],cr_image.shape[1])
            
            # Üç farklı ölçek kullan: merkez, sol-üst ve sağ-alt rakıma göre düzelt
            rotated_image = cuda_resize_if_available(cr_image, (int(width*olcek_scale_test), int(height*olcek_scale_test)), interpolation=cv2.INTER_NEAREST)
            
            olcek_scale_sol_ust=olcek_scale_test*(rakim_sol_ust/rakim)
            olcek_scale_sag_alt=olcek_scale_test*(rakim_sag_alt/rakim)
            
            #olcek_scale_sol_ust=olcek_scale_test*(rakim/rakim_sol_ust)
            #olcek_scale_sag_alt=olcek_scale_test*(rakim/rakim_sag_alt)
            
            rotated_image_sol_ust = cuda_resize_if_available(cr_image, (int(width*olcek_scale_sol_ust), int(height*olcek_scale_sol_ust)), interpolation=cv2.INTER_NEAREST)
            rotated_image_sag_alt = cuda_resize_if_available(cr_image, (int(width*olcek_scale_sag_alt), int(height*olcek_scale_sag_alt)), interpolation=cv2.INTER_NEAREST)
            
            
            #çözünürlüğü 30 cm'ye ayarlanmış görüntünün orta noktası bulnur
            height, width = rotated_image.shape[:2]
            # get the center coordinates of the image to create the 2D rotation matrix
            center = (int(width/2), int(height/2))
            
            fark=np.minimum(center[0],center[1])-272    # 544'lık frame'in elde edilen dikdörtgenin dışına taşmaması için yazıldı 
            if fark>200:
                fark=200
            elif fark<0:
                print("merkezi dışarıda")
                continue
           
            
            y1 = center[1]-PATCH_HALF-fark; y2 = center[1]+PATCH_HALF-fark
            x1 = center[0]-PATCH_HALF-fark; x2 = center[0]+PATCH_HALF-fark
            if not is_valid_slice(rotated_image_sol_ust, x1, y1, x2, y2):
                print("rotated_part1 sınır dışında")
                continue
            rotated_part1 = rotated_image_sol_ust[y1:y2, x1:x2]
            y1 = center[1]-PATCH_HALF; y2 = center[1]+PATCH_HALF
            x1 = center[0]-PATCH_HALF; x2 = center[0]+PATCH_HALF
            if not is_valid_slice(rotated_image, x1, y1, x2, y2):
                print("rotated_part2 sınır dışında")
                continue
            rotated_part2 = rotated_image[y1:y2, x1:x2]
            y1 = center[1]-PATCH_HALF+fark; y2 = center[1]+PATCH_HALF+fark
            x1 = center[0]-PATCH_HALF+fark; x2 = center[0]+PATCH_HALF+fark
            if not is_valid_slice(rotated_image_sag_alt, x1, y1, x2, y2):
                print("rotated_part3 sınır dışında")
                continue
            rotated_part3 = rotated_image_sag_alt[y1:y2, x1:x2]
            
            
            
            # cv2.imshow('Original image', image)
            # cv2.imshow('Rotated image', rotated_image)
            if DEBUG:
                cv2.imshow('Rotated part', rotated_part2)
                _ = cv2.waitKey(1) 
            
            
            # 7.2) Template listesi (3 ölçek)
            template=[]
            
            template.append(rotated_part1)
            template.append(rotated_part2)
            template.append(rotated_part3)
            ######################################################################################################
            
            # template = anlik_goruntu
            # template = cv2.imread(template,0)
            # plt.imshow(template, cmap = "gray")
            
            
            
            # model_yolu=model_yol+model_list[k]
            
            # model = load_model(model_yolu)
            
            # Batch preprocess and predict for the 3 templates
            # 7.3) Model giriş ön işlemleri (resize/equalize/normalize)
            pre_list = []
            for j in range(3):
                t_resized = cv2.resize(template[j], (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
                t_resized = cv2.equalizeHist(t_resized)
                t_resized = ((t_resized.astype(np.float32) - 127.5) / 127.5)
                pre_list.append(t_resized)

            batch = np.stack(pre_list, axis=0)[..., None]
            pred = model.predict(batch)
            if pred.ndim == 4:
                pred = pred.squeeze(axis=-1)

            for j in range(3):
                out = pred[j]
                out = (out * 255.0).astype('uint8')
                out = out[PRED_BORDER:PATCH_SIZE-PRED_BORDER, PRED_BORDER:PATCH_SIZE-PRED_BORDER]
                template[j] = out
                
            print(template[0].shape)
            h,w =template[0].shape
                
                #plt.imshow(template[j], cmap = "gray")
            if DEBUG:
                cv2.imshow("model uygulanmis", template[1])
                _ = cv2.waitKey(1) 
                
            
            # gdal.Warp('anlik_goruntu_warped.tif', anlik_goruntu, xRes=0.09, yRes=0.09) 
            # raster = gdal.Open('anlik_goruntu_warped.tif')
            # gt =raster.GetGeoTransform()
            
            # print (gt)
            # pixelSizeX = gt[1]
            # pixelSizeY = -gt[5]
            # print ("x = ",pixelSizeX)
            # print ("y = ",pixelSizeY)
            
            
            #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
            
            
            #paralel programlama ile aynı anda 3 templatematching yapılır
            inputs=[(cerceve,template[0]),(cerceve,template[1]),(cerceve,template[2])]
            # Template matching (sequential to avoid IPC overhead)
            res1, res2, res3 = match_three(cerceve, [template[0], template[1], template[2]])
            # Not: CUDA varsa tek seferde görüntü yüklenip üç eşleşme GPU’da yapılır; aksi halde CPU.
            #methods =['cv2.TM_CCOEFF']
            #for meth in methods:
                #method  = eval(meth)    #stringleri fonksiyona çeviren fonksiyona
                # res1= cv2.matchTemplate(img, template[0], method, None)
                # res2= cv2.matchTemplate(img, template[1], method, None)
                # res3= cv2.matchTemplate(img, template[2], method, None)
            print(res1.shape)
            min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
            min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
            min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
                
            print(min_val2, max_val2, min_loc2, max_loc2)
                
                # if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
                #     top_left1 =min_loc1
                # else:
                    
            top_left1 = (max_loc1[0] + konum[0]-int(cerceve.shape[0]/2),max_loc1[1] + konum[1]-int(cerceve.shape[0]/2))
            top_left2 = (max_loc2[0] + konum[0]-int(cerceve.shape[0]/2),max_loc2[1] + konum[1]-int(cerceve.shape[0]/2))
            top_left3 = (max_loc3[0] + konum[0]-int(cerceve.shape[0]/2),max_loc3[1] + konum[1]-int(cerceve.shape[0]/2))
            
            
         
            bottom_right1 = (top_left1[0] + w,top_left1[1] + h)
            bottom_right2 = (top_left2[0] + w,top_left2[1] + h)
            bottom_right3 = (top_left3[0] + w,top_left3[1] + h)
            
                #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                
            # global_x = max_loc2[0]+int(w/2)
            # global_y = max_loc2[1]+int(h/2)
                 
            
                 
            # Üç aday dikdörtgenin (x,y,w,h) biçiminde paketlenmesi
            a=(top_left1[0],top_left1[1],w,h)
            b=(top_left2[0],top_left2[1],w,h)
            c=(top_left3[0],top_left3[1],w,h)
            
            uzaklik_ab=math.dist(a, b)
            uzaklik_bc=math.dist(b, c) 
            uzaklik_ac=math.dist(a, c) 
            
            
            
            if (uzaklik_ab+uzaklik_bc-uzaklik_ac)<2 and benchmark==False:
                cerceve_boyutu=cerceve_boyutu_deger
            else:
                cerceve_boyutu+=100
                
                 
                 #konum bulmak için kesişimler ve kesişim karelerinin koordinatları bulunuyor
            kesisim_ab = intersection(a, b);
            kesisim_bc = intersection(b, c);
            kesisim_ac = intersection(a, c);
                
                
            if kesisim_ab!=() and kesisim_bc!=() and kesisim_ac!=():
                kesisim_abc=intersection(kesisim_ab, kesisim_bc)
                kare=(kesisim_abc[0],kesisim_abc[1],int(kesisim_abc[2]),int(kesisim_abc[3]))
                print("konum: ",kare)
                cerceve_boyutu+=100
                yanlis_pozitif_kontrol+=1
                dogru_pozitif_kontrol+=1
                
            elif kesisim_ab!=() :
                kare=(kesisim_ab[0],kesisim_ab[1],int(kesisim_ab[2]),int(kesisim_ab[3]))
                print("konum: ",kare)
                cerceve_boyutu+=100
                yanlis_pozitif_kontrol+=1
                dogru_pozitif_kontrol+=1
            elif kesisim_bc!=() :
                kare=(kesisim_bc[0],kesisim_bc[1],int(kesisim_bc[2]),int(kesisim_bc[3]))
                print("konum: ",kare)
                cerceve_boyutu+=100
                yanlis_pozitif_kontrol+=1
                dogru_pozitif_kontrol+=1
            elif kesisim_ac!=() :
                kare=(kesisim_ac[0],kesisim_ac[1],int(kesisim_ac[2]),int(kesisim_ac[3]))
                print("konum: ",kare)
                cerceve_boyutu+=100
                yanlis_pozitif_kontrol+=1
                dogru_pozitif_kontrol+=1
                        
            else:
                print("kesişim yok")
                kare=(0,0,0,0)
                kare=b
                cerceve_boyutu+=500
                yanlis_negatif_kontrol+=1
                
            
            
            
            
            # Kesişim merkezinin koordinatı (piksel cinsinden)
            konum_y=kare[0]+int(kare[2]/2)
            konum_x=kare[1]+int(kare[3]/2)
                
            if konum_y>img.shape[1]:
                konum_y=img.shape[1]-1
            if konum_x>img.shape[0]:
                konum_x=img.shape[0]-1                
                     
            
            
            konum=(konum_y,konum_x)
            
           
            
            
            
            
            #konum = (kare[0]+int(kare[2]/2),kare[1]+int(kare[3]/2))
            
            """
                gps_longtidye ce gps_latitde değişkenleri anlık görüntünün korrdinatlarını verir
                koordinatlar[1][konum[0]][konum[1]] ise modelin tahmin ettiği konumun koordinatlarını verir
                ve aralarındaki uzaklık hesaplanır.    
            """
            long_tahmin, lat_tahmin = rc_to_ll(konum[1], konum[0])
            
                
            uzaklik = haversine_distance(gps_latitude,gps_longitude,lat_tahmin,long_tahmin)  
            #uzaklik2 = haversine_distance(gps_latitude,gps_longitude,lat_tahmin,long_tahmin)  
            print("uzaklik = {:.3f} km".format(uzaklik)) 
            
            uzaklik_hatalari.append(uzaklik)

                
            # if(uzaklik<=0.05):
            #     dogru_tahmin+=1
            #     sonuclar.append([[anlik_yol_list[i]],["Dogru"],[(gps_latitude,gps_longitude)],[(lat_tahmin,long_tahmin)]])
                    
            # else:
            #     yanlis_tahmin+=1
            #     sonuclar.append([[anlik_yol_list[i]],["Yanlis"],[(gps_latitude,gps_longitude)],[(lat_tahmin,long_tahmin)]])
            
            
            
            # Başarı eşiği: 70 metre (0.07 km). Duruma göre TP/FP/TN/FN sayaçları güncellenir.
            if(uzaklik<=0.07):
                if yanlis_negatif_kontrol>0:
                    yanlis_negatif+=1
                else:
                    dogru_pozitif+=1
                    
                dogru_tahmin+=1
                sonuclar.append([[anlik_yol_list[i]],["Dogru"],[gps_latitude],[gps_longitude],[lat_tahmin],[long_tahmin],[ucus_yuksekligi]])
                
                    
            else:
                if yanlis_pozitif_kontrol>0:
                    yanlis_pozitif+=1
                else:
                    dogru_negatif+=1
                    
                yanlis_tahmin+=1
                sonuclar.append([[anlik_yol_list[i]],["Yanlis"],[gps_latitude],[gps_longitude],[lat_tahmin],[long_tahmin],[ucus_yuksekligi]])
               
                
               
            if uzaklik>0.3 and benchmark==False:
                konum=konum_once
               
          
                
            
                
            # dosyaya_yaz(sonuclar,dogru_tahmin,yanlis_tahmin)  # Döngü sonunda bir defa yazılacak
                
            centerOfCircle=konum    
                    
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            if benchmark==True:
                cerceve_boyutu=cerceve_boyutu_deger
                konum=(knm[1],knm[0])
                
            
            cv2.rectangle(img, (-int(cerceve_boyutu/2)+konum[0],-int(cerceve_boyutu/2)+konum[1]), (+int(cerceve_boyutu/2)+konum[0],+int(cerceve_boyutu/2)+konum[1]),(0,0,0),25)

            cv2.rectangle(img, top_left1, bottom_right1,(0,0,255),25)
            cv2.rectangle(img, top_left2, bottom_right2,(0,255,0),25)
            cv2.rectangle(img, top_left3, bottom_right3,(255,0,0),25)
            radius=10
            cv2.circle(img, centerOfCircle, radius, (0,255,255), 25)   #tahmini konumu veren nokta
            cv2.circle(img,(knm[1],knm[0]),radius,(0,255,0), 25)                   #gerçek konumu gösteren nokta
                    #plt.figure()
                    
                
                    # plt.imshow(img)
                    # plt.title("Tespit edilen Sonuç"), plt.axis("on")
                    # plt.suptitle(meth)
                    # plt.pause(0.0001)
            #res = cv2.resize(img, dsize=(766*2,1595*2), interpolation=cv2.INTER_CUBIC)
                #cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
                
                
            ressol=-3000+knm[0]
            ressag=+3000+knm[0]
            resust=-3000+knm[1]
            resalt=+3000+knm[1]
            
            if ressol<0:
                ressol=0
            if ressag<0:
                ressag= 0
            if resust<0:
                resust= 0
            if resalt<0:
                resalt= 0
                
            
            # Uçak simgesini (gerçek konum ve heading ile) çiz
            try:
                draw_plane_icon_v2(img, (knm[1], knm[0]), yaw, size_px=220, color=(255,0,255), outline=(0,0,0), outline_thickness=10)
            except Exception:
                pass

            res = img[ressol:ressag,resust:resalt] 
                
                
                
            window_name = 'Image'

            # 7.4) HUD: başlık (yaw), uçuş yüksekliği ve hatayı göster; ölçek çubuğu ve hedef işaretleri çiz
            hud_lines = [
                f"HDG: {yaw:.1f} deg",
                f"ALT: {int(ucus_yuksekligi)} m",
                f"ERR: {int(uzaklik*1000)} m",
            ]
            draw_info_panel(
                res,
                hud_lines,
                top_left=(25, 150),
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=6,
                thickness=20,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                alpha=0.5,
                padding=25,
            )

            # Add a 100 m scale bar if spatial resolution is known (cm/px)
            try:
                draw_scale_bar(res, mekansal_cozunurluk, scale_meters=100, margin=80, bar_height=40,
                               color=(255,255,255), text_color=(255,255,255), font_scale=3, thickness=8)
            except Exception:
                pass

            # Subtle center crosshair for orientation
            try:
                cv2.drawMarker(res, (res.shape[1]//2, res.shape[0]//2), (255,255,255),
                               markerType=cv2.MARKER_CROSS, markerSize=120, thickness=8, line_type=cv2.LINE_AA)
            except Exception:
                pass
                
                
                
                
                
            cv2.namedWindow("konum", cv2.WINDOW_NORMAL)  
            cv2.resizeWindow("konum", 1000, 1000)
            cv2.imshow("konum", res)
            _ = cv2.waitKey(1)   #☺ekrana verilen haritayı anlık görebilmek için yazılır
            
                # cv2.rectangle(img, top_left, bottom_right,(255,0,0),35)
                # plt.figure()
                # plt.subplot(121), plt.imshow(res, cmap = "gray")
                # plt.title("Eşleşen Sonuç"), plt.axis("on")
                # plt.subplot(122), plt.imshow(img)
                # plt.title("Tespit edilen Sonuç"), plt.axis("on")
                # plt.suptitle(meth)
                # img = cv2.imread(harita,0)
        
        
            bitis_zamani = time.time()
            calisma_suresi = bitis_zamani - baslangic_zamani
            #print("Kodun calisma suresi___________________________________________:", calisma_suresi, "saniye")
            
            print("Kodun calisma suresi___________________________________________:", "{:.2f}".format(calisma_suresi), "saniye")
            

            print((i+1),"/",(len(anlik_yol_list)),"     dogru_tahmin: ,"+str(dogru_tahmin)+",  yanlis_tahmin: ,"+str(yanlis_tahmin) +",  dogru pozitif: "+str(dogru_pozitif)+",  yanlış pozitif: "+str(yanlis_pozitif)+",  dogru negatif: "+str(dogru_negatif)+",  yanlış negatif: "+str(yanlis_negatif)+"\n")
         
        
        # 8) Döngü sonu: kaynakları serbest bırak, hata metriklerini hesapla ve çıktı dosyalarına yaz
        # Close map dataset for this loop to free resources
        try:
            map_ds.close()
        except Exception:
            pass
        uzaklik_hatalari = np.array(uzaklik_hatalari)

        rmse_degeri = rmse(uzaklik_hatalari)
        mae_degeri = mae(uzaklik_hatalari)
        standart_sapma_degeri = standart_sapma(uzaklik_hatalari)
        
        
        # Yeni verilen değerler için tekrar hesaplama yapılıyor


        
        # Hassasiyet ve Geri Çağırma hesaplamaları
        hassasiyet_yeni = dogru_pozitif / (dogru_pozitif + yanlis_pozitif)
        geri_cagirma_yeni = dogru_pozitif / (dogru_pozitif + yanlis_negatif)
        
        # F Skoru hesaplama
        f_skoru = 2 * (hassasiyet_yeni * geri_cagirma_yeni) / (hassasiyet_yeni + geri_cagirma_yeni)
        

        
        sonuclar_dosya = open("modele_gore_sonuclar.txt", "a+")

       # sonuclar = np.vstack((sonuclar,dogru_tahmin, yanlis_tahmin)).T
       # print(sonuclar)
        sonuclar_=anlik_yol[-20:]+" "+str(model_list[k])+",  dogru_tahmin: ,"+str(dogru_tahmin)+",  yanlis_tahmin: ,"+str(yanlis_tahmin) + ",  RMSE Değeri: ,"+str(rmse_degeri)+ ",  MAE Değeri: ,"+str(mae_degeri)+",  standart sapma Değeri: ,"+str(standart_sapma_degeri)+",  dogru pozitif: ,"+str(dogru_pozitif)+",  yanlış pozitif: ,"+str(yanlis_pozitif)+",  dogru negatif: ,"+str(dogru_negatif)+",  yanlış negatif: ,"+str(yanlis_negatif)+", f skoru: ,"+str(f_skoru)+"\n"
     
        sonuclar_dosya.write(sonuclar_)
        sonuclar_dosya.close()
      
            
        print("dogru tahmin = ",dogru_tahmin)
        print("yanlış tahmin = ",yanlis_tahmin)
        print("yanlış pozitif = ",yanlis_pozitif)
        yuzde=dogru_tahmin/(dogru_tahmin+yanlis_tahmin)
        yuzde=yuzde*100
        print("doğruluk yüzdesi: {:.2f}".format(yuzde))
        
        # Döngü sonu: sonuçları bir defa yaz
        try:
            dosyaya_yaz(sonuclar, dogru_tahmin, yanlis_tahmin)
        except Exception as _e:
            print("sonuclar dosyaya yazılırken hata:", _e)

        try:
            dem_ds.close()
        except Exception:
            pass
        try:
            del dataset
        except Exception:
            pass
        input("pause")
