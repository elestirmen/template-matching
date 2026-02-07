"""
Bu betik, IHA/drone goruntulerini referans harita uzerinde konumlandirir.

Ozet akis:
- EXIF: yaw, GPS, focal length ve altitude okunur.
- DEM: yukseklik farkina gore 3 farkli olcek uretilir.
- Keras ciktilari template matching icin kullanilir.
- En iyi 3 eslesmenin kesisiminden konum bulunur.
- RMSE/MAE/std ve siniflandirma metrikleri yazdirilir.

Klasorler:
- haritalar/: ana harita rasterlari
- model/: keras model dosyalari
- parcalar/: anlik goruntu klasoru
"""
# -----------------------------------------------------------------------------
# RUN_CFG: Tum calisma parametrelerini bu bloktan yonetebilirsiniz.
# Bu bolumu dosyanin en ustunde tutuyoruz ki erisim kolay olsun.
# -----------------------------------------------------------------------------
RUN_CFG = {
    # Genel calisma modu
    "BENCHMARK": False,
    "DEBUG": False,

    # Model/patch ayarlari
    "PATCH_SIZE": 544,
    "PRED_BORDER": 16,

    # Template matching hizlandirma
    "USE_PYRAMID": True,
    "COARSE_SCALE": 0.5,
    "ROI_PAD_FACTOR": 0.4,

    # Arama cercevesi boyutu
    "CERCEVE_BOYUTU_NORMAL": 2048,
    "CERCEVE_BOYUTU_BENCHMARK": 5000,

    # Veri yollari
    "HARITA_DIR": "haritalar",
    "MODEL_DIR": "model",
    "ANLIK_DIR": "parcalar",
    "DEM_PATH": "ana_harita_urgup_30_cm_utm_elevation.tif",

    # Isterseniz dogrudan dosya secin (bos birakirsaniz klasordeki tum dosyalar kullanilir)
    "HARITA_DOSYALARI": [],  # ornek: ["map1.tif", "map2.tif"]
    "MODEL_DOSYALARI": [],   # ornek: ["m1.h5", "m2.h5"]
    "SORT_INPUTS": False,

    # EXIF/camera fallback
    "DEFAULT_FOCAL_LENGTH_MM": 8.8,
    "DEFAULT_SENSOR_WIDTH_MM": 13.2,
    "USE_GPS_ALT_REF_SIGN": False,

    # Trajectory cizimi (tahmin=sari, gercek=yesil)
    "DRAW_TRAJECTORY": True,
    "TRAJECTORY_DRAW_POINTS": True,
    "TRAJECTORY_MAX_POINTS": 0,   # 0: sinirsiz
    "TRAJECTORY_LINE_THICKNESS": 15,
    "TRAJECTORY_POINT_RADIUS": 20,

    # Runtime UI butonlari (konum penceresinde tikla-ac/kapa)
    "UI_BUTTONS_ENABLED": True,
    "UI_BUTTON_FONT_SCALE": 1.0,
    "UI_BUTTON_THICKNESS": 2,
    "UI_BUTTON_SCALE": 0.5,
    "UI_WINDOW_WIDTH": 1000,
    "UI_WINDOW_HEIGHT": 1000,
    "SHOW_INNER_FRAME": True,
    "SHOW_TM_BOXES": True,

    # Calisma sonu bekleme
    "WAIT_PER_MODEL": False,
    "WAIT_ON_EXIT": False,
}

benchmark = bool(RUN_CFG["BENCHMARK"])
DEBUG = bool(RUN_CFG["DEBUG"])
PATCH_SIZE = int(RUN_CFG["PATCH_SIZE"])
PATCH_HALF = PATCH_SIZE // 2
PRED_BORDER = int(RUN_CFG["PRED_BORDER"])
USE_PYRAMID = bool(RUN_CFG["USE_PYRAMID"])
COARSE_SCALE = float(RUN_CFG["COARSE_SCALE"])
ROI_PAD_FACTOR = float(RUN_CFG["ROI_PAD_FACTOR"])
DRAW_TRAJECTORY = bool(RUN_CFG.get("DRAW_TRAJECTORY", False))
TRAJECTORY_DRAW_POINTS = bool(RUN_CFG.get("TRAJECTORY_DRAW_POINTS", True))
TRAJECTORY_MAX_POINTS = int(RUN_CFG.get("TRAJECTORY_MAX_POINTS", 0))
TRAJECTORY_LINE_THICKNESS = int(RUN_CFG.get("TRAJECTORY_LINE_THICKNESS", 10))
TRAJECTORY_POINT_RADIUS = int(RUN_CFG.get("TRAJECTORY_POINT_RADIUS", 8))
UI_BUTTONS_ENABLED = bool(RUN_CFG.get("UI_BUTTONS_ENABLED", True))
UI_BUTTON_FONT_SCALE = float(RUN_CFG.get("UI_BUTTON_FONT_SCALE", 1.0))
UI_BUTTON_THICKNESS = int(RUN_CFG.get("UI_BUTTON_THICKNESS", 3))
UI_BUTTON_SCALE = float(RUN_CFG.get("UI_BUTTON_SCALE", 1.0))
UI_WINDOW_WIDTH = int(RUN_CFG.get("UI_WINDOW_WIDTH", 1280))
UI_WINDOW_HEIGHT = int(RUN_CFG.get("UI_WINDOW_HEIGHT", 960))
SHOW_INNER_FRAME = bool(RUN_CFG.get("SHOW_INNER_FRAME", True))
SHOW_TM_BOXES = bool(RUN_CFG.get("SHOW_TM_BOXES", True))

if benchmark:
    cerceve_boyutu_deger = int(RUN_CFG["CERCEVE_BOYUTU_BENCHMARK"])
else:
    cerceve_boyutu_deger = int(RUN_CFG["CERCEVE_BOYUTU_NORMAL"])

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
dirname = os.path.dirname(os.path.abspath(__file__))

def _get_screen_size():
    """Return (screen_width, screen_height) or a safe fallback."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        try:
            import tkinter as tk
            root = tk.Tk(); root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            try:
                root.destroy()
            except Exception:
                pass
            return int(w), int(h)
        except Exception:
            return 1920, 1080

def _show_image_fit(win_name, img, max_frac=0.95):
    """Show image in a resizable window sized to fit the screen while keeping aspect ratio."""
    try:
        h, w = img.shape[:2]
        sw, sh = _get_screen_size()
        scale = min(1.0, (sw * float(max_frac)) / float(w), (sh * float(max_frac)) / float(h))
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        except Exception:
            pass
        cv2.resizeWindow(win_name, disp_w, disp_h)
        cv2.imshow(win_name, img)
    except Exception:
        try:
            cv2.imshow(win_name, img)
        except Exception:
            pass


def _compose_side_by_side(left_gray, right_gray, left_title="Sol", right_title="SaÃ„Å¸",
                          target_height=900, apply_colormap_right=True):
    """Create a labeled side-by-side visualization from two grayscale images.

    - Resizes both images to the same target height while preserving aspect.
    - Converts to BGR and adds titles on top-left.
    - Optionally applies a colormap to the right image to emphasize details.
    Returns a BGR image suitable for cv2.imshow.
    """
    try:
        if left_gray is None or right_gray is None:
            return None

        # Ensure 2D grayscale arrays
        if left_gray.ndim == 3:
            if left_gray.shape[2] == 3:
                left_gray = cv2.cvtColor(left_gray, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_gray.squeeze()
        if right_gray.ndim == 3:
            if right_gray.shape[2] == 3:
                right_gray = cv2.cvtColor(right_gray, cv2.COLOR_BGR2GRAY)
            else:
                right_gray = right_gray.squeeze()

        # Convert to 8-bit if needed
        if left_gray.dtype != np.uint8:
            lmin, lmax = float(np.min(left_gray)), float(np.max(left_gray))
            if lmax > lmin:
                left_gray = np.clip((left_gray - lmin) * (255.0 / (lmax - lmin)), 0, 255).astype(np.uint8)
            else:
                left_gray = np.zeros_like(left_gray, dtype=np.uint8)
        if right_gray.dtype != np.uint8:
            rmin, rmax = float(np.min(right_gray)), float(np.max(right_gray))
            if rmax > rmin:
                right_gray = np.clip((right_gray - rmin) * (255.0 / (rmax - rmin)), 0, 255).astype(np.uint8)
            else:
                right_gray = np.zeros_like(right_gray, dtype=np.uint8)

        # Resize to target height with preserved aspect ratio
        def _resize_to_h(img, H):
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                return None
            new_w = max(1, int(w * (H / float(h))))
            return cv2.resize(img, (new_w, H), interpolation=cv2.INTER_AREA)

        target_height = int(target_height)
        target_height = max(200, min(2000, target_height))
        L = _resize_to_h(left_gray, target_height)
        R = _resize_to_h(right_gray, target_height)
        if L is None or R is None:
            return None

        # Optional colormap on right
        if apply_colormap_right:
            try:
                Rc = cv2.applyColorMap(R, cv2.COLORMAP_VIRIDIS)
            except Exception:
                Rc = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
        else:
            Rc = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)

        Lc = cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)

        # Add simple titles
        def _title(img, text):
            try:
                cv2.putText(img, str(text), (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(img, str(text), (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 8, cv2.LINE_AA)
            except Exception:
                pass
            return img

        Lc = _title(Lc, left_title)
        Rc = _title(Rc, right_title)

        try:
            vis = cv2.hconcat([Lc, Rc])
        except Exception:
            # Fallback manual concat if hconcat unavailable
            H = max(Lc.shape[0], Rc.shape[0])
            W = Lc.shape[1] + Rc.shape[1]
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            vis[:Lc.shape[0], :Lc.shape[1]] = Lc
            vis[:Rc.shape[0], Lc.shape[1]:Lc.shape[1]+Rc.shape[1]] = Rc
        return vis
    except Exception:
        return None


def _compose_top_bottom(top_gray, bottom_gray, top_title="Crop", bottom_title="Model",
                        target_width=900, apply_colormap_bottom=True,
                        caption_height=80, gap=10):
    try:
        if top_gray is None or bottom_gray is None:
            return None

        # Normalize shapes: keep top as color if provided; ensure bottom is single-channel
        top_is_color = (top_gray.ndim == 3 and top_gray.shape[2] == 3)
        if not top_is_color and top_gray.ndim == 3:
            top_gray = top_gray.squeeze()
        if bottom_gray.ndim == 3:
            if bottom_gray.shape[2] == 3:
                bottom_gray = cv2.cvtColor(bottom_gray, cv2.COLOR_BGR2GRAY)
            else:
                bottom_gray = bottom_gray.squeeze()

        # Convert to 8-bit if needed
        if top_gray.dtype != np.uint8:
            if top_is_color:
                tmin, tmax = float(np.min(top_gray)), float(np.max(top_gray))
                if tmax > tmin:
                    top_gray = np.clip((top_gray - tmin) * (255.0 / (tmax - tmin)), 0, 255).astype(np.uint8)
                else:
                    top_gray = np.zeros_like(top_gray, dtype=np.uint8)
            else:
                tmin, tmax = float(np.min(top_gray)), float(np.max(top_gray))
                if tmax > tmin:
                    top_gray = np.clip((top_gray - tmin) * (255.0 / (tmax - tmin)), 0, 255).astype(np.uint8)
                else:
                    top_gray = np.zeros_like(top_gray, dtype=np.uint8)
        if bottom_gray.dtype != np.uint8:
            bmin, bmax = float(np.min(bottom_gray)), float(np.max(bottom_gray))
            if bmax > bmin:
                bottom_gray = np.clip((bottom_gray - bmin) * (255.0 / (bmax - bmin)), 0, 255).astype(np.uint8)
            else:
                bottom_gray = np.zeros_like(bottom_gray, dtype=np.uint8)

        # Resize to same target width
        def _resize_to_w(img, W):
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                return None
            new_h = max(1, int(h * (W / float(w))))
            return cv2.resize(img, (W, new_h), interpolation=cv2.INTER_AREA)

        target_width = int(max(200, min(3000, target_width)))
        T = _resize_to_w(top_gray, target_width)
        B = _resize_to_w(bottom_gray, target_width)
        if T is None or B is None:
            return None

        # Optional colormap on bottom
        if apply_colormap_bottom:
            try:
                Bc = cv2.applyColorMap(B, cv2.COLORMAP_VIRIDIS)
            except Exception:
                Bc = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
        else:
            Bc = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
        if top_is_color and T.ndim == 3 and T.shape[2] == 3:
            Tc = T
        else:
            Tc = cv2.cvtColor(T, cv2.COLOR_GRAY2BGR)

        # Create caption bars (text outside the image)
        cap_h = int(max(30, min(200, caption_height)))
        W = target_width
        top_bar = np.zeros((cap_h, W, 3), dtype=np.uint8)
        bottom_bar = np.zeros((cap_h, W, 3), dtype=np.uint8)

        def _draw_caption(bar, text):
            try:
                cv2.putText(bar, str(text), (25, int(cap_h*0.75)), cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(bar, str(text), (25, int(cap_h*0.75)), cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, (0, 0, 0), 8, cv2.LINE_AA)
            except Exception:
                pass
            return bar

        top_bar = _draw_caption(top_bar, top_title)
        bottom_bar = _draw_caption(bottom_bar, bottom_title)

        # Assemble with optional gap between images
        gap_h = int(max(0, min(200, gap)))
        gap_bar = np.zeros((gap_h, W, 3), dtype=np.uint8) if gap_h > 0 else None

        top_tile = cv2.vconcat([top_bar, Tc])
        bottom_tile = cv2.vconcat([bottom_bar, Bc])

        if gap_bar is not None:
            vis = cv2.vconcat([top_tile, gap_bar, bottom_tile])
        else:
            vis = cv2.vconcat([top_tile, bottom_tile])
        return vis
    except Exception:
        return None

# -----------------------------------------------------------------------------
# YardÃ„Â±mcÃ„Â± fonksiyonlar gruplarÃ„Â±
# - CUDA kontrolÃƒÂ¼ ve hÃ„Â±zlandÃ„Â±rÃ„Â±lmÃ„Â±Ã…Å¸ iÃ…Å¸lemler (resize / template matching)
# - EXIF/GPS okuma ve dÃƒÂ¶nÃƒÂ¼Ã…Å¸ÃƒÂ¼mler (WGS84 <-> UTM, piksel <-> koordinat)
# - Basit geometri ve metrikler (kesiÃ…Å¸im, RMSE/MAE/std, Haversine)
# - GÃƒÂ¶rsel arayÃƒÂ¼z yardÃ„Â±mcÃ„Â±larÃ„Â± (HUD paneli, ÃƒÂ¶lÃƒÂ§ek ÃƒÂ§ubuÃ„Å¸u, iÃ…Å¸aret ÃƒÂ§izimi)
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
    
    # Veri ÃƒÂ§erÃƒÂ§evesini oluÃ…Å¸tur
    df = pd.DataFrame(sonuclar, columns=['goruntu', 'sonuc', 'gercek_latitude', 'gercek_longitude', 'tahmini_latitude', 'tahmini_longitude','ucus_yuksekligi'])
    
    # EÃ„Å¸er her bir hÃƒÂ¼cre bir liste iÃƒÂ§eriyorsa, bu listelerin ilk elemanÃ„Â±nÃ„Â± al
    for column in df.columns:
        df[column] = df[column].apply(lambda x: x[0] if isinstance(x, list) else x)

    # Metin dosyasÃ„Â±na yaz
    with open("sonuclar.txt", "w") as sonuclar_dosya:
        sonuclar_dosya.write(df.to_string())
    
    # CSV dosyasÃ„Â±na kaydet
    df.to_csv("sonuclar.csv", index=False)


    
#exif bilgisi okur    
def get_field (exif,field) :
  for (k,v) in exif.items():
     if TAGS.get(k) == field:
        return v
 
 #gos coordinatÃ„Â±nÃ„Â± decimal sisteme ÃƒÂ§evirir
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
        m = re.search(r'FlightDegree[^0-9+-]*([+-]?\d+(?:\.\d+)?)', s)
        if m:
            try:
                yaw = float(m.group(1)) / 10.0
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
            if isinstance(ref, bytes):
                ref = ref.decode(errors='ignore')
            if isinstance(ref, str):
                ref = ref.strip().upper()
            else:
                ref = str(ref).strip().upper()
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
    if isinstance(alt_ref, (bytes, bytearray)) and len(alt_ref) > 0:
        alt_ref = int(alt_ref[0])
    # Bazi DJI veri setlerinde GPSAltitudeRef guvenilir degil olabiliyor.
    # Varsayilan olarak isareti ters cevirmiyoruz; ihtiyac olursa env ile ac.
    if altitude is not None and alt_ref == 1 and bool(RUN_CFG.get("USE_GPS_ALT_REF_SIGN", False)):
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
# ---------------------------------------------------------------------------
# UI renk paleti -- tum UI elemanlari bu paleti kullanir
# ---------------------------------------------------------------------------
UI_COLORS = {
    "panel_bg":       (30, 30, 40),
    "panel_border":   (80, 80, 100),
    "btn_on":         (76, 175, 80),
    "btn_off":        (70, 75, 90),
    "btn_hover_on":   (100, 200, 105),
    "btn_hover_off":  (90, 95, 115),
    "toggle_on":      (76, 175, 80),
    "toggle_off":     (120, 120, 130),
    "toggle_knob":    (255, 255, 255),
    "text_primary":   (240, 240, 245),
    "text_secondary": (170, 175, 190),
    "text_shadow":    (0, 0, 0),
    "accent":         (66, 133, 244),
    "header_bg":      (45, 48, 65),
    "collapse_btn":   (55, 60, 80),
    "collapse_hover": (80, 85, 110),
}


def _draw_alpha_panel(img, x0, y0, x1, y1, color=(0, 0, 0), alpha=0.5):
    """Draw a filled rectangle with alpha blending onto img in-place (ROI-only)."""
    x0 = max(0, min(int(x0), img.shape[1] - 1))
    x1 = max(0, min(int(x1), img.shape[1] - 1))
    y0 = max(0, min(int(y0), img.shape[0] - 1))
    y1 = max(0, min(int(y1), img.shape[0] - 1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x1 - x0, y1 - y0), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _draw_rounded_rect(img, x0, y0, x1, y1, radius, color, thickness=-1):
    """Yuvarlatilmis koseli dikdortgen cizer."""
    r = max(0, min(int(radius), (x1 - x0) // 2, (y1 - y0) // 2))
    if r < 2:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        return
    # Iki dikdortgen + 4 kose dairesi
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, thickness)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, thickness)
    cv2.circle(img, (x0 + r, y0 + r), r, color, thickness)
    cv2.circle(img, (x1 - r, y0 + r), r, color, thickness)
    cv2.circle(img, (x0 + r, y1 - r), r, color, thickness)
    cv2.circle(img, (x1 - r, y1 - r), r, color, thickness)


def _draw_alpha_rounded_panel(img, x0, y0, x1, y1, radius, color=(0, 0, 0), alpha=0.5):
    """Yuvarlatilmis koseli yari-seffaf panel cizer (ROI-only)."""
    x0 = max(0, min(int(x0), img.shape[1] - 1))
    x1 = max(0, min(int(x1), img.shape[1] - 1))
    y0 = max(0, min(int(y0), img.shape[0] - 1))
    y1 = max(0, min(int(y1), img.shape[0] - 1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    _draw_rounded_rect(overlay, 0, 0, x1 - x0, y1 - y0, radius, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _draw_toggle_switch(img, x, y, w, h, is_on, view_scale=1.0):
    """iOS tarzi toggle switch cizer."""
    r = h // 2
    bg = UI_COLORS["toggle_on"] if is_on else UI_COLORS["toggle_off"]
    knob = UI_COLORS["toggle_knob"]
    # Kapsul arka plan
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), bg, -1)
    cv2.circle(img, (x + r, y + r), r, bg, -1)
    cv2.circle(img, (x + w - r, y + r), r, bg, -1)
    # Beyaz top (knob)
    knob_pad = max(3, int(round(4 * view_scale)))
    if is_on:
        cx = x + w - r - knob_pad
    else:
        cx = x + r + knob_pad
    cv2.circle(img, (cx, y + r), r - knob_pad, knob, -1)
    # Ince border
    border_t = max(1, int(round(1.5 * view_scale)))
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), (200, 200, 210), border_t)
    cv2.circle(img, (x + r, y + r), r, (200, 200, 210), border_t)
    cv2.circle(img, (x + w - r, y + r), r, (200, 200, 210), border_t)


def _draw_text_with_shadow(img, text, pos, font, scale, color, thickness,
                            shadow_color=None, shadow_offset=2):
    """Golge efektli metin cizer."""
    if shadow_color is None:
        shadow_color = UI_COLORS["text_shadow"]
    so = max(1, int(shadow_offset))
    cv2.putText(img, text, (pos[0] + so, pos[1] + so), font, scale,
                shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_info_panel(img, lines, top_left=(25, 150), font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=6, thickness=20, text_color=None,
                    bg_color=None, alpha=0.55, padding=25, line_gap=None,
                    corner_radius=0):
    """Draw a semi-transparent info panel with multiple text lines.

    lines: list of strings to show, one per line
    top_left: baseline of the first text line (x, y)
    corner_radius: >0 ise rounded panel cizer
    """
    if not lines:
        return
    if text_color is None:
        text_color = UI_COLORS["text_primary"]
    if bg_color is None:
        bg_color = UI_COLORS["panel_bg"]
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
    panel_x1 = min(img.shape[1] - 1, panel_x0 + panel_w)
    panel_y1 = min(img.shape[0] - 1, panel_y0 + panel_h)
    cr = int(corner_radius)
    if cr > 2:
        _draw_alpha_rounded_panel(img, panel_x0, panel_y0, panel_x1, panel_y1,
                                  radius=cr, color=bg_color, alpha=alpha)
        border_t = max(1, thickness // 8)
        _draw_rounded_rect(img, panel_x0, panel_y0, panel_x1, panel_y1, cr,
                           UI_COLORS["panel_border"], border_t)
    else:
        _draw_alpha_panel(img, panel_x0, panel_y0, panel_x1, panel_y1,
                          color=bg_color, alpha=alpha)

    for i, s in enumerate(lines):
        org = (int(x), int(y + i * line_gap))
        _draw_text_with_shadow(img, str(s), org, font, font_scale,
                               text_color, thickness)


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


def _build_runtime_buttons():
    """Return clickable button definitions for runtime visibility toggles."""
    return [
        {"key": "_panel_collapsed", "label": "", "hotkey": "H",
         "rect": (20, 20, 44, 44), "is_collapse": True},
        {"key": "trajectory",   "label": "Trajektori",   "hotkey": "T", "rect": (20, 20, 260, 54)},
        {"key": "inner_frame",  "label": "Ic Cerceve",   "hotkey": "I", "rect": (20, 84, 260, 54)},
        {"key": "tm_boxes",     "label": "TM RGB Kutu",  "hotkey": "R", "rect": (20, 148, 260, 54)},
    ]


def _draw_runtime_buttons(
    img,
    ui_state,
    buttons,
    font_scale=1.0,
    thickness=3,
    ui_scale=1.0,
    display_size=(1280, 960),
):
    """Draw modern, adaptive ON/OFF toggle buttons onto the current frame.

    Ozellikler:
    - Rounded-corner panel ve butonlar
    - iOS tarzi toggle switch
    - Hover efekti (fare ile ustune gelince renk degisir)
    - Collapse/expand (panel kucultme/buyutme)
    - Tutarli renk paleti (UI_COLORS)
    """
    if img is None or not buttons:
        return
    try:
        h_img, w_img = img.shape[:2]
    except Exception:
        return

    try:
        dw = max(1, int(display_size[0]))
        dh = max(1, int(display_size[1]))
    except Exception:
        dw, dh = 1280, 960

    view_scale = max(float(w_img) / float(dw), float(h_img) / float(dh), 1.0)
    view_scale *= max(0.25, float(ui_scale))

    margin = max(12, int(round(20 * view_scale)))
    gap = max(6, int(round(10 * view_scale)))
    bw = max(220, int(round(360 * view_scale)))
    bh = max(44, int(round(64 * view_scale)))
    panel_pad = max(10, int(round(16 * view_scale)))
    header_h = max(30, int(round(42 * view_scale)))
    bw = min(bw, max(160, w_img - 2 * margin))
    corner_r = max(6, int(round(12 * view_scale)))
    btn_r = max(4, int(round(8 * view_scale)))

    # Toggle switch boyutlari
    toggle_w = max(50, int(round(80 * view_scale)))
    toggle_h = max(22, int(round(32 * view_scale)))
    toggle_margin = max(8, int(round(12 * view_scale)))

    x0 = margin
    y0 = margin

    is_collapsed = bool(ui_state.get("_panel_collapsed", False))
    hover_key = ui_state.get("_hover_key", None)

    # -- Sadece collapse butonu olan toggle butonunu bul --
    collapse_btn = None
    content_buttons = []
    for b in buttons:
        if b.get("is_collapse"):
            collapse_btn = b
        else:
            content_buttons.append(b)

    # ==================== COLLAPSED gorunum ====================
    if is_collapsed:
        cb_size = max(36, int(round(50 * view_scale)))
        cb_x0 = x0
        cb_y0 = y0
        cb_x1 = cb_x0 + cb_size
        cb_y1 = cb_y0 + cb_size

        is_cb_hover = (hover_key == "_panel_collapsed")
        cb_fill = UI_COLORS["collapse_hover"] if is_cb_hover else UI_COLORS["collapse_btn"]

        _draw_alpha_rounded_panel(img, cb_x0, cb_y0, cb_x1, cb_y1,
                                  radius=btn_r, color=UI_COLORS["panel_bg"], alpha=0.65)
        _draw_rounded_rect(img, cb_x0, cb_y0, cb_x1, cb_y1, btn_r, cb_fill, thickness=-1)
        _draw_rounded_rect(img, cb_x0, cb_y0, cb_x1, cb_y1, btn_r,
                           UI_COLORS["panel_border"],
                           max(1, int(round(2 * view_scale))))

        if collapse_btn is not None:
            collapse_btn["rect"] = (cb_x0, cb_y0, cb_size, cb_size)

        # Hamburger ikonu (3 yatay cizgi)
        bar_w = int(cb_size * 0.5)
        bar_h = max(2, int(round(3 * view_scale)))
        bar_x = cb_x0 + (cb_size - bar_w) // 2
        bar_gap = max(4, int(round(6 * view_scale)))
        bar_y_center = cb_y0 + cb_size // 2
        for dy in (-bar_gap, 0, bar_gap):
            by = bar_y_center + dy - bar_h // 2
            cv2.rectangle(img, (bar_x, by), (bar_x + bar_w, by + bar_h),
                          UI_COLORS["text_primary"], -1)
        return

    # ==================== EXPANDED gorunum ====================
    panel_w = bw + (2 * panel_pad)
    panel_h = (header_h + (2 * panel_pad) +
               (len(content_buttons) * bh) +
               (max(0, len(content_buttons) - 1) * gap))

    px0 = x0 - panel_pad
    py0 = y0 - panel_pad
    px1 = px0 + panel_w
    py1 = py0 + panel_h

    # Panel arka plan (rounded, yari-seffaf)
    _draw_alpha_rounded_panel(img, px0, py0, px1, py1,
                              radius=corner_r, color=UI_COLORS["panel_bg"], alpha=0.60)
    _draw_rounded_rect(img, px0, py0, px1, py1, corner_r,
                       UI_COLORS["panel_border"],
                       max(1, int(round(2 * view_scale))))

    # -- Header bolumu --
    hdr_y1 = py0 + header_h + panel_pad
    _draw_alpha_rounded_panel(img, px0, py0, px1, hdr_y1,
                              radius=corner_r, color=UI_COLORS["header_bg"], alpha=0.35)

    title_scale = max(0.55, float(font_scale) * 0.75 * view_scale)
    title_thick = max(1, int(round(float(thickness) * 0.8 * view_scale)))
    title_x = x0 + int(round(8 * view_scale))
    title_y = y0 + int(round(header_h * 0.6))
    _draw_text_with_shadow(img, "GORUNUM", (title_x, title_y),
                           cv2.FONT_HERSHEY_SIMPLEX, title_scale,
                           UI_COLORS["accent"], title_thick, shadow_offset=2)

    # -- Collapse butonu (header sag ust) --
    if collapse_btn is not None:
        cb_size = max(28, int(round(36 * view_scale)))
        cb_x0 = px1 - cb_size - max(4, int(round(6 * view_scale)))
        cb_y0_c = py0 + max(4, int(round(6 * view_scale)))
        cb_x1 = cb_x0 + cb_size
        cb_y1 = cb_y0_c + cb_size

        is_cb_hover = (hover_key == "_panel_collapsed")
        cb_fill = UI_COLORS["collapse_hover"] if is_cb_hover else UI_COLORS["collapse_btn"]

        _draw_rounded_rect(img, cb_x0, cb_y0_c, cb_x1, cb_y1, btn_r // 2,
                           cb_fill, thickness=-1)
        _draw_rounded_rect(img, cb_x0, cb_y0_c, cb_x1, cb_y1, btn_r // 2,
                           UI_COLORS["panel_border"],
                           max(1, int(round(1.5 * view_scale))))

        collapse_btn["rect"] = (cb_x0, cb_y0_c, cb_size, cb_size)

        # Minimize ikonu (yatay cizgi)
        line_w = int(cb_size * 0.5)
        line_h = max(2, int(round(3 * view_scale)))
        lx = cb_x0 + (cb_size - line_w) // 2
        ly = cb_y0_c + cb_size // 2 - line_h // 2
        cv2.rectangle(img, (lx, ly), (lx + line_w, ly + line_h),
                      UI_COLORS["text_primary"], -1)

    # -- Butonlar --
    y = y0 + header_h
    txt_scale = max(0.6, float(font_scale) * 0.78 * view_scale)
    txt_thick = max(1, int(round(float(thickness) * 0.85 * view_scale)))

    for b in content_buttons:
        key = b["key"]
        is_on = bool(ui_state.get(key, False))
        is_hovered = (hover_key == key)
        label = str(b.get("label", key))
        hotkey = str(b.get("hotkey", "")).strip()

        # Renk secimi (hover duyarli)
        if is_on:
            fill = UI_COLORS["btn_hover_on"] if is_hovered else UI_COLORS["btn_on"]
        else:
            fill = UI_COLORS["btn_hover_off"] if is_hovered else UI_COLORS["btn_off"]

        edge = UI_COLORS["accent"] if is_hovered else UI_COLORS["panel_border"]

        b["rect"] = (x0, y, bw, bh)

        # Buton arka plani (rounded)
        _draw_rounded_rect(img, x0, y, x0 + bw, y + bh, btn_r, fill, thickness=-1)
        _draw_rounded_rect(img, x0, y, x0 + bw, y + bh, btn_r, edge,
                           max(1, int(round(2 * view_scale))))

        # Sol: label + hotkey
        label_text = f"{label}  [{hotkey}]" if hotkey else label
        lx = x0 + max(10, int(round(14 * view_scale)))
        ly = y + int(round(bh * 0.62))
        _draw_text_with_shadow(img, label_text, (lx, ly),
                               cv2.FONT_HERSHEY_SIMPLEX, txt_scale,
                               UI_COLORS["text_primary"], txt_thick)

        # Sag: toggle switch
        tw_x = x0 + bw - toggle_w - toggle_margin
        tw_y = y + (bh - toggle_h) // 2
        _draw_toggle_switch(img, tw_x, tw_y, toggle_w, toggle_h,
                            is_on, view_scale)

        y += (bh + gap)


def _runtime_buttons_mouse_cb(event, x, y, flags, userdata):
    """Mouse callback for runtime toggle buttons (hover + click + collapse).

    NOT: OpenCV WINDOW_NORMAL modunda mouse koordinatlarini otomatik olarak
    goruntu piksel koordinatlarina cevirir, bu yuzden ek donusum gerekmez.
    """
    if not isinstance(userdata, dict):
        return
    ui_state = userdata.get("state")
    buttons = userdata.get("buttons", [])
    if not isinstance(ui_state, dict):
        return

    # OpenCV zaten goruntu koordinatlarini verir -- dogrudan kullan
    mx, my = x, y

    # -- Hover tracking (her mouse hareketi) --
    if event == cv2.EVENT_MOUSEMOVE:
        ui_state["_hover_key"] = None
        for b in buttons:
            try:
                bx, by, bw, bh = b["rect"]
            except (ValueError, KeyError):
                continue
            if bx <= mx <= (bx + bw) and by <= my <= (by + bh):
                ui_state["_hover_key"] = b["key"]
                break
        return

    # -- Click handling --
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for b in buttons:
        try:
            bx, by, bw, bh = b["rect"]
        except (ValueError, KeyError):
            continue
        if bx <= mx <= (bx + bw) and by <= my <= (by + bh):
            k = b["key"]
            ui_state[k] = not bool(ui_state.get(k, False))
            break







def draw_plane_icon_v2(img, center, heading_deg, size_px=200,
                       color=(255, 0, 255), outline=(0, 0, 0), outline_thickness=8):
    """Harita uzerinde konum ve yon gosteren temiz bir ucak silueti cizer.

    - center: (x, y) piksel koordinati
    - heading_deg: yaw/heading (derece). 0=Kuzey, pozitif saat yonu.
    - size_px: burun-kuyruk uzunlugu (piksel).
    """
    try:
        cx, cy = int(center[0]), int(center[1])

        # PNG varsa onu kullan
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
            ang = float(heading_deg) - 90.0
            ih, iw = icon.shape[:2]
            M = cv2.getRotationMatrix2D((iw/2.0, ih/2.0), ang, 1.0)
            abs_cos = abs(M[0,0]); abs_sin = abs(M[0,1])
            nW = int(ih * abs_sin + iw * abs_cos)
            nH = int(ih * abs_cos + iw * abs_sin)
            M[0,2] += (nW/2.0) - iw/2.0
            M[1,2] += (nH/2.0) - ih/2.0
            rot = cv2.warpAffine(icon, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            scale = float(size_px) / max(nW, nH)
            if scale != 1.0:
                rot = cv2.resize(rot, (max(1,int(nW*scale)), max(1,int(nH*scale))), interpolation=cv2.INTER_AREA)
            rh, rw = rot.shape[:2]
            x1 = cx - rw//2; y1 = cy - rh//2
            x2 = x1 + rw;   y2 = y1 + rh
            H, W = img.shape[:2]
            rx1 = max(0, x1); ry1 = max(0, y1)
            rx2 = min(W, x2); ry2 = min(H, y2)
            if rx1 < rx2 and ry1 < ry2:
                sx1 = rx1 - x1; sy1 = ry1 - y1
                sx2 = sx1 + (rx2 - rx1); sy2 = sy1 + (ry2 - ry1)
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

        # --- Vektorel ucak silueti ---
        L = float(size_px)
        if L <= 0:
            return

        def _clamp_rgb(col):
            if isinstance(col, (list, tuple, np.ndarray)):
                vals = list(col)
            else:
                vals = [col, col, col]
            while len(vals) < 3:
                vals.append(0)
            return tuple(int(np.clip(float(v), 0, 255)) for v in vals[:3])

        base_color  = _clamp_rgb(color)
        outline_rgb = _clamp_rgb(outline if outline is not None else (0, 0, 0))

        # Donusum matrisi
        ang = float(heading_deg)
        rad = np.deg2rad(ang)
        co, si = np.cos(rad), np.sin(rad)
        R_mat = np.array([[co, -si], [si, co]], dtype=np.float32)

        def _xf(pts_list):
            pts = np.array(pts_list, dtype=np.float32)
            if pts.size == 0:
                return None
            pts_r = pts @ R_mat.T
            pts_r[:, 0] += cx
            pts_r[:, 1] += cy
            return pts_r.astype(np.int32)

        # Boyut oranlari
        fw = L * 0.10   # govde yarim genisligi
        ws = L * 0.60   # kanat yari acikligi
        ts = L * 0.26   # kuyruk yari acikligi

        # Tum ucak konturu (simetrik, sade)
        plane = [
            # Burun
            [0.0,       -0.50 * L],
            [fw * 0.5,  -0.42 * L],
            [fw,        -0.28 * L],
            # Sag kanat
            [fw,        -0.06 * L],
            [ws,         0.02 * L],
            [ws * 0.95,  0.10 * L],
            [fw,         0.12 * L],
            # Govde orta-arka
            [fw * 0.85,  0.30 * L],
            # Sag kuyruk
            [ts,         0.42 * L],
            [ts * 0.90,  0.50 * L],
            [fw * 0.35,  0.48 * L],
            # Kuyruk ucu
            [0.0,        0.52 * L],
            # Sol kuyruk
            [-fw * 0.35, 0.48 * L],
            [-ts * 0.90, 0.50 * L],
            [-ts,        0.42 * L],
            # Govde sol
            [-fw * 0.85, 0.30 * L],
            [-fw,        0.12 * L],
            [-ws * 0.95, 0.10 * L],
            [-ws,        0.02 * L],
            [-fw,       -0.06 * L],
            # Sol govde-burun
            [-fw,       -0.28 * L],
            [-fw * 0.5, -0.42 * L],
        ]

        plane_xf = _xf(plane)
        if plane_xf is None:
            return

        # Dolgu
        cv2.fillPoly(img, [plane_xf], base_color, lineType=cv2.LINE_AA)

        # Dis hat
        if outline_thickness > 0:
            cv2.polylines(img, [plane_xf], isClosed=True, color=outline_rgb,
                          thickness=max(1, outline_thickness), lineType=cv2.LINE_AA)

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

# Parametreleri asagidaki RUN_CFG bolumunden ayarlayin.
# -----------------------------------------------------------------------------
# RUN_CFG: Tum calisma parametrelerini bu bloktan yonetebilirsiniz.
# Burada degistirdiginiz degerler tum script'e uygulanir.
# -----------------------------------------------------------------------------
RUN_CFG = {
    # Genel calisma modu
    "BENCHMARK": False,
    "DEBUG": False,

    # Model/patch ayarlari
    "PATCH_SIZE": 544,
    "PRED_BORDER": 16,

    # Template matching hizlandirma
    "USE_PYRAMID": True,
    "COARSE_SCALE": 0.5,
    "ROI_PAD_FACTOR": 2.0,

    # Arama cercevesi boyutu
    "CERCEVE_BOYUTU_NORMAL": 2048,
    "CERCEVE_BOYUTU_BENCHMARK": 5000,

    # Veri yollari
    "HARITA_DIR": "haritalar",
    "MODEL_DIR": "model",
    "ANLIK_DIR": "parcalar",
    "DEM_PATH": "ana_harita_urgup_30_cm_utm_elevation.tif",

    # Isterseniz dogrudan dosya secin (bos birakirsaniz klasordeki tum dosyalar kullanilir)
    "HARITA_DOSYALARI": [],  # ornek: ["map1.tif", "map2.tif"]
    "MODEL_DOSYALARI": [],   # ornek: ["m1.h5", "m2.h5"]
    "SORT_INPUTS": False,

    # EXIF/camera fallback
    "DEFAULT_FOCAL_LENGTH_MM": 8.8,
    "DEFAULT_SENSOR_WIDTH_MM": 13.2,
    "USE_GPS_ALT_REF_SIGN": False,

    # Calisma sonu bekleme
    "WAIT_PER_MODEL": False,
    "WAIT_ON_EXIT": False,
}

benchmark = bool(RUN_CFG["BENCHMARK"])
DEBUG = bool(RUN_CFG["DEBUG"])
PATCH_SIZE = int(RUN_CFG["PATCH_SIZE"])
PATCH_HALF = PATCH_SIZE // 2
PRED_BORDER = int(RUN_CFG["PRED_BORDER"])
USE_PYRAMID = bool(RUN_CFG["USE_PYRAMID"])
COARSE_SCALE = float(RUN_CFG["COARSE_SCALE"])
ROI_PAD_FACTOR = float(RUN_CFG["ROI_PAD_FACTOR"])

if benchmark:
    cerceve_boyutu_deger = int(RUN_CFG["CERCEVE_BOYUTU_BENCHMARK"])
else:
    cerceve_boyutu_deger = int(RUN_CFG["CERCEVE_BOYUTU_NORMAL"])



#%%

from math import radians, sin, cos, sqrt, atan2

def calculate_coordinates(latitude, longitude, d_lat, d_long):
    R = 6378137  # YarÃ„Â±ÃƒÂ§apÃ„Â± metre cinsinden olan WGS-84 elipsoiti
    
    new_latitude = latitude + (d_lat / R) * (180 / 3.14159265358979323846)
    new_longitude = longitude + (d_long / (R * cos(3.14159265358979323846 * latitude / 180))) * (180 / 3.14159265358979323846)
    
    return new_latitude, new_longitude

def find_corner_coordinates(center_latitude, center_longitude, pixel_distance, GSD):
    # GSD (Ground Sample Distance): Metre cinsinden piksel baÃ…Å¸Ã„Â±na dÃƒÂ¼Ã…Å¸en gerÃƒÂ§ek dÃƒÂ¼nya uzunluÃ„Å¸u
    # pixel_distance: KaÃƒÂ§ piksel uzaklÃ„Â±kta yeni bir nokta oluÃ…Å¸turulacaÃ„Å¸Ã„Â±
    
    distance = pixel_distance * GSD  # Metre cinsinden toplam mesafe
    
    # Sol ÃƒÂ¼st kÃƒÂ¶Ã…Å¸e koordinatlarÃ„Â±
    new_latitude1, new_longitude1 = calculate_coordinates(center_latitude, center_longitude, -distance, -distance)
    
    # SaÃ„Å¸ alt kÃƒÂ¶Ã…Å¸e koordinatlarÃ„Â±
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
            print("CUDA template matching kullanÃ„Â±lamadÃ„Â±, CPU'ya dÃƒÂ¼Ã…Å¸ÃƒÂ¼lÃƒÂ¼yor:", e)
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

            # Coarse downscale (local search seed)
            s = COARSE_SCALE
            small_W = max(1, int(W * s))
            small_H = max(1, int(H * s))
            small_w = max(1, int(w * s))
            small_h = max(1, int(h * s))

            # If coarse scale is not feasible, keep search local around center.
            if small_w > small_W or small_h > small_H:
                cx = resW // 2
                cy = resH // 2
            else:
                a_small = cv2.resize(a, (small_W, small_H), interpolation=cv2.INTER_AREA)
                b_small = cv2.resize(b, (small_w, small_h), interpolation=cv2.INTER_AREA)
                res_small = cv2.matchTemplate(a_small, b_small, method, None)
                if res_small.size == 0:
                    cx = resW // 2
                    cy = resH // 2
                else:
                    _, _, _, max_loc_small = cv2.minMaxLoc(res_small)
                    # Map back to full-res match coordinates
                    cx = int(max_loc_small[0] / s) if s > 0 else (resW // 2)
                    cy = int(max_loc_small[1] / s) if s > 0 else (resH // 2)

            # ROI around coarse location in res-space
            pad = max(8, int(max(w, h) * ROI_PAD_FACTOR))
            x1 = max(0, cx - pad)
            y1 = max(0, cy - pad)
            x2 = min(resW - 1, cx + pad)
            y2 = min(resH - 1, cy + pad)

            # Ensure at least one candidate position; do not fall back to full search.
            if x2 < x1:
                x1 = x2 = max(0, min(cx, resW - 1))
            if y2 < y1:
                y1 = y2 = max(0, min(cy, resH - 1))

            # Corresponding image region for refined match
            img_x1 = x1
            img_y1 = y1
            img_x2 = x2 + w - 1
            img_y2 = y2 + h - 1

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
        print("CUDA TM kullanÃ„Â±lmadÃ„Â± (hata), CPU'ya dÃƒÂ¼Ã…Å¸ÃƒÂ¼lÃƒÂ¼yor:", e)
        return tuple(cv2.matchTemplate(img, t, method, None) for t in templates)
    
# RMSE hesaplama fonksiyonu
def rmse(errors):
    squared_errors = errors ** 2                     # hatalarÃ„Â±n karesini al
    mean_squared_errors = squared_errors.mean()     # karelerin ortalamasÃ„Â±nÃ„Â± al
    rmse_val = np.sqrt(mean_squared_errors)          # OrtalamanÃ„Â±n karekÃƒÂ¶kÃƒÂ¼nÃƒÂ¼ al
    return rmse_val


# MAE hesaplama fonksiyonu
def mae(errors):
    absolute_errors = np.abs(errors)                # hatalarÃ„Â±n mutlak deÃ„Å¸erini al
    mean_absolute_errors = absolute_errors.mean()   # mutlak hatalarÃ„Â±n ortalamasÃ„Â±nÃ„Â± al
    return mean_absolute_errors

def standart_sapma(data):
    mean = np.mean(data)                     # OrtalamayÃ„Â± hesapla
    squared_diff = (data - mean) ** 2        # Ortalama ile farklarÃ„Â±n karesini al
    mean_squared_diff = np.mean(squared_diff)  # Kare farklarÃ„Â±nÃ„Â±n ortalamasÃ„Â±nÃ„Â± al
    std_dev = np.sqrt(mean_squared_diff)     # OrtalamanÃ„Â±n karekÃƒÂ¶kÃƒÂ¼nÃƒÂ¼ al
    return std_dev





#%%





def _normalize_ext_set(exts):
    """Return normalized lowercase extension set like {'.jpg', '.h5'}."""
    out = set()
    for e in (exts or []):
        if not isinstance(e, str):
            continue
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.add(e)
    return out


def _ext_allowed(path, allowed_exts):
    if not allowed_exts:
        return True
    return os.path.splitext(path)[1].lower() in allowed_exts


def _list_files_filtered(folder, allowed_exts):
    """List files in a folder filtered by extension; ignore unrelated files."""
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            continue
        if not _ext_allowed(p, allowed_exts):
            continue
        files.append(p)
    return files


def _filter_candidates(paths, allowed_exts, label):
    """Validate explicit file candidates, skipping missing or wrong-extension files."""
    kept = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"Uyari: {label} dosyasi bulunamadi, atlandi: {p}")
            continue
        if not _ext_allowed(p, allowed_exts):
            print(f"Uyari: {label} uzantisi desteklenmiyor, atlandi: {p}")
            continue
        kept.append(p)
    return kept


if __name__ == '__main__': 
    # Log CUDA environment once
    try:
        log_cuda_info_once()
    except Exception:
        pass
    
    # 1) Yol/klasor hazirligi: harita, model, anlik goruntu ve DEM
    harita_yol = RUN_CFG["HARITA_DIR"]
    if not os.path.isabs(harita_yol):
        harita_yol = os.path.join(dirname, harita_yol)

    model_yol = RUN_CFG["MODEL_DIR"]
    if not os.path.isabs(model_yol):
        model_yol = os.path.join(dirname, model_yol)

    anlik_yol = RUN_CFG["ANLIK_DIR"]
    if not os.path.isabs(anlik_yol):
        anlik_yol = os.path.join(dirname, anlik_yol)

    if not os.path.isdir(harita_yol):
        raise RuntimeError(f"Harita klasoru bulunamadi: {harita_yol}")
    if not os.path.isdir(model_yol):
        raise RuntimeError(f"Model klasoru bulunamadi: {model_yol}")
    if not os.path.isdir(anlik_yol):
        raise RuntimeError(f"Anlik goruntu klasoru bulunamadi: {anlik_yol}")

    # Desteklenen uzantilar: alakasiz dosyalari (.gitkeep vb.) otomatik dislar.
    harita_exts = _normalize_ext_set(
        RUN_CFG.get("HARITA_UZANTILARI", [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".jp2"])
    )
    model_exts = _normalize_ext_set(
        RUN_CFG.get("MODEL_UZANTILARI", [".h5", ".keras", ".hdf5"])
    )
    anlik_exts = _normalize_ext_set(
        RUN_CFG.get("ANLIK_UZANTILARI", [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"])
    )

    harita_secili = RUN_CFG.get("HARITA_DOSYALARI", [])
    if harita_secili:
        harita_aday_list = [
            (p if os.path.isabs(p) else os.path.join(harita_yol, p))
            for p in harita_secili
        ]
        harita_path_list = _filter_candidates(harita_aday_list, harita_exts, "harita")
    else:
        harita_path_list = _list_files_filtered(harita_yol, harita_exts)

    model_secili = RUN_CFG.get("MODEL_DOSYALARI", [])
    if model_secili:
        model_aday_list = [
            (p if os.path.isabs(p) else os.path.join(model_yol, p))
            for p in model_secili
        ]
        model_path_list = _filter_candidates(model_aday_list, model_exts, "model")
    else:
        model_path_list = _list_files_filtered(model_yol, model_exts)

    if bool(RUN_CFG.get("SORT_INPUTS", False)):
        harita_path_list = sorted(harita_path_list)
        model_path_list = sorted(model_path_list)

    if not harita_path_list:
        raise RuntimeError(
            f"Harita dosyasi bulunamadi (uzantilar: {sorted(harita_exts)}): {harita_yol}"
        )
    if not model_path_list:
        raise RuntimeError(
            f"Model dosyasi bulunamadi (uzantilar: {sorted(model_exts)}): {model_yol}"
        )

    if len(harita_path_list) != len(model_path_list):
        print(
            f"Uyari: harita/model sayisi farkli (harita={len(harita_path_list)}, model={len(model_path_list)}). "
            f"Ilk {min(len(harita_path_list), len(model_path_list))} cift kullanilacak."
        )
    eslesme_sayisi = min(len(harita_path_list), len(model_path_list))

    model_list = [os.path.basename(p) for p in model_path_list]
    harita_yol_list = [os.path.basename(p) for p in harita_path_list]

    ana_harita_elevation = RUN_CFG["DEM_PATH"]
    anlik_yol_list = _list_files_filtered(anlik_yol, anlik_exts)
    try:
        anlik_yol_list = sorted(anlik_yol_list, key=lambda p: os.path.getmtime(p))
    except Exception:
        anlik_yol_list = sorted(anlik_yol_list)
    if not anlik_yol_list:
        raise RuntimeError(
            f"Anlik goruntu dosyasi bulunamadi (uzantilar: {sorted(anlik_exts)}): {anlik_yol}"
        )
    
    
    
    
    
    # haritadaki piksellerin gps koordinatlarÃ„Â± bulunur ve koordinatlar olarak ayrÃ„Â± bri dosya olarak diske kaydedilir. bir kez ÃƒÂ§alÃ„Â±Ã…Å¸tÃ„Â±rÃ„Â±lmasÃ„Â± yeterlidir
    ###############################################################################
    #%%
    
    import rasterio
    from affine import Affine
 
    #fname = 'urgup_gmap_georef.tif'
    fname = harita_path_list[0]
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
    
    #DEM verileri aktarÃ„Â±lÃ„Â±r
    
    # 3) DEM rasterÃ„Â±nÃ„Â± (elevation) aÃƒÂ§
    filename = ana_harita_elevation
    if not os.path.isabs(filename):
        filename = os.path.join(dirname, filename)
            
    dataset = gdal.Open(filename)
    
    gt = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)  #5. bant elevation bandÃ„Â±
    
    DEM_array = band.ReadAsArray()
    # RasterIO dataset and transformer for DEM pixel lookup
    dem_ds = rio.open(filename)
    ll_to_dem = Transformer.from_crs("EPSG:4326", dem_ds.crs, always_xy=True)
    
    ###############################################################################
    
    #%%
    
    
    # 4) BaÃ…Å¸langÃ„Â±ÃƒÂ§ durumlarÃ„Â± ve toplayÃ„Â±cÃ„Â±lar
    cerceve_boyutu=cerceve_boyutu_deger
    sonuclar = []
    
    konum=(0,0)
    konum_once=(0,0)
    kare=()  
    runtime_ui_state = {
        "trajectory": bool(DRAW_TRAJECTORY),
        "inner_frame": bool(SHOW_INNER_FRAME),
        "tm_boxes": bool(SHOW_TM_BOXES),
        "_panel_collapsed": False,
        "_hover_key": None,
    }
    runtime_ui_buttons = _build_runtime_buttons() if UI_BUTTONS_ENABLED else []
    runtime_ui_ctx = {
        "state": runtime_ui_state,
        "buttons": runtime_ui_buttons,
        "display_size": (UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT),
        "img_size": None,
    }
    runtime_ui_cb_set = False
    
    
    
    
    for k in range(eslesme_sayisi):
        yanlis_pozitif=0
        dogru_pozitif=0
        dogru_negatif=0
        yanlis_negatif=0
        uzaklik_hatalari = []
        # Her model/harita cifti icin gecmis izleri ayri tutulur.
        traj_pred_points = []
        traj_real_points = []
        
        model_yolu = model_path_list[k]
        model = load_model(model_yolu)        
        
        
          
        dogru_tahmin=0
        yanlis_tahmin=0
        ana_harita = harita_path_list[k]
        # Referans haritayÃ„Â± gri-ton olarak oku (Template Matching iÃƒÂ§in daha uygundur)
          
        t_img = cv2.imread(ana_harita,0)  #haritalar klasÃƒÂ¶rÃƒÂ¼ndeki ikinci gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼yÃƒÂ¼ okur
        if t_img is None:
            print("Harita okunamadi, atlaniyor:", ana_harita)
            continue
        print(t_img.shape)

        # Open main map once per loop and reuse for pixel lookups
        map_ds = rio.open(ana_harita)
        ll_to_map = Transformer.from_crs("EPSG:4326", map_ds.crs, always_xy=True)
        rc_to_ll = make_rc_to_ll(map_ds)
          
        kenarx=int(t_img.shape[0]/512)
        
        # 6) Anlik goruntu dosyalari (filtrelenmis liste yukarida bir kez hazirlandi)
        uzaklik=0
        fark=100
        irtifa_dizisi=[]
        # 7) Her anlÃ„Â±k gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼ iÃƒÂ§in dÃƒÂ¶ngÃƒÂ¼
        for i in range(len(anlik_yol_list)):
            
            yanlis_pozitif_kontrol = 0            
            dogru_pozitif_kontrol = 0
            dogru_negatif_kontrol  = 0
            yanlis_negatif_kontrol = 0
            
            baslangic_zamani = time.time()

            
            konum_once=konum
            
            
            img = t_img
            
            print(img.shape)
            anlik_goruntu = anlik_yol_list[i]
            if not os.path.exists(anlik_goruntu):
                print("dosya bulunamadÃ„Â±:", anlik_goruntu)
                continue
            exif_data = parse_exif(anlik_goruntu)
            if exif_data is None:
                print("EXIF okunamadi, atlaniyor:", anlik_goruntu)
                continue

            yaw = float(exif_data.get('yaw') or 0.0)
            gps_latitude = exif_data.get('latitude')
            gps_longitude = exif_data.get('longitude')
            altitude = exif_data.get('altitude')
            FocalLength = exif_data.get('focal_length')
            kamera_model = exif_data.get('model')

            if gps_latitude is None or gps_longitude is None:
                print("GPS EXIF eksik, atlaniyor:", anlik_goruntu)
                continue
            if altitude is None:
                print("Altitude EXIF eksik, atlaniyor:", anlik_goruntu)
                continue

            if FocalLength is None or float(FocalLength) <= 0:
                try:
                    FocalLength = float(RUN_CFG.get("DEFAULT_FOCAL_LENGTH_MM", 8.8))
                    print("Uyari: FocalLength EXIF eksik/gecersiz, varsayilan deger kullanildi:", FocalLength)
                except Exception:
                    print("FocalLength elde edilemedi, atlaniyor:", anlik_goruntu)
                    continue

            try:
                with Image.open(anlik_goruntu) as im_meta:
                    goruntu_piksel_genisligi, goruntu_piksel_yuksekligi = im_meta.size
            except Exception:
                print("Goruntu boyutu okunamadi, atlaniyor:", anlik_goruntu)
                continue
           
            
           
            # Use pre-opened map and transformer for fast lookups
            # Harita ÃƒÂ¼zerinde EXIF koordinatÃ„Â±na karÃ…Å¸Ã„Â±lÃ„Â±k gelen pikseli bul
            knm = piksel_bul_fast(map_ds, ll_to_map, gps_longitude, gps_latitude)
            

            
            
                       
                
            # Ã„Â°lk karede EXIF konumuna yakÃ„Â±n ÃƒÂ§evrede, sonraki karelerde bir ÃƒÂ¶nceki tahmine yakÃ„Â±n ÃƒÂ§evrede ara
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
            
            
            
            
            sol = max(0, min(sol, img.shape[0]))
            sag = max(0, min(sag, img.shape[0]))
            ust = max(0, min(ust, img.shape[1]))
            alt = max(0, min(alt, img.shape[1]))
            if sag <= sol or alt <= ust:
                print("cerceve gecersiz, atlaniyor")
                continue
            cerceve = img[sol:sag,ust:alt]
            
            
            
            if knm[0]<272 or knm[0]>img.shape[0]-272:
                print("dÃ„Â±Ã…Å¸arÃ„Â±da")
                continue
            elif knm[1]<272 or knm[1]>img.shape[1]-272:
                print("dÃ„Â±Ã…Å¸arÃ„Â±da")
                continue
            
            
            
            sol_ust, sag_alt = find_corner_coordinates(gps_latitude, gps_longitude, 100, 0.30)            
            
            #anlÃ„Â±k gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼nÃƒÂ¼n ana haritada karÃ…Å¸Ã„Â±lÃ„Â±k geldiÃ„Å¸i rakÃ„Â±m deÃ„Å¸eri bulunur
            
            dem_konum = piksel_bul_fast(dem_ds, ll_to_dem, gps_longitude, gps_latitude)
            
            dem_konum_sol_ust = piksel_bul_fast(dem_ds, ll_to_dem, sol_ust[1], sol_ust[0])
            dem_konum_sag_alt = piksel_bul_fast(dem_ds, ll_to_dem, sag_alt[1], sag_alt[0])

            
            def _in_dem_bounds(rc):
                return 0 <= int(rc[0]) < DEM_array.shape[0] and 0 <= int(rc[1]) < DEM_array.shape[1]

            if not (_in_dem_bounds(dem_konum) and _in_dem_bounds(dem_konum_sol_ust) and _in_dem_bounds(dem_konum_sag_alt)):
                print("DEM disinda kalan koordinat, atlaniyor")
                continue

            rakim = float(DEM_array[dem_konum[0],dem_konum[1]])
            rakim_sol_ust = float(DEM_array[dem_konum_sol_ust[0],dem_konum_sol_ust[1]])
            rakim_sag_alt = float(DEM_array[dem_konum_sag_alt[0],dem_konum_sag_alt[1]])

            if (not np.isfinite(rakim)) or (not np.isfinite(rakim_sol_ust)) or (not np.isfinite(rakim_sag_alt)):
                print("DEM degeri gecersiz, atlaniyor")
                continue

            print(rakim)

            rakim_duzeltme = 26
            camera_sensor_by_model = {
                "L1D-20c": 13.2,  # Mavic 2 Pro
                "FC2204": 6.17,   # Mavic 2 Zoom
            }
            camera_sensor_genislik = camera_sensor_by_model.get(kamera_model)
            if camera_sensor_genislik is None:
                camera_sensor_genislik = float(RUN_CFG.get("DEFAULT_SENSOR_WIDTH_MM", 13.2))
                print(f"Uyari: bilinmeyen kamera modeli ({kamera_model}), sensor genisligi fallback: {camera_sensor_genislik} mm")

            camera_focal_lenght = float(FocalLength)
            if camera_focal_lenght <= 0:
                print("FocalLength gecersiz, atlaniyor:", camera_focal_lenght)
                continue

            if goruntu_piksel_genisligi <= 0:
                print("Goruntu piksel genisligi gecersiz, atlaniyor")
                continue

            ucus_yuksekligi = altitude - rakim + rakim_duzeltme
            if ucus_yuksekligi <= 0:
                print("Ucus yuksekligi gecersiz, atlaniyor:", ucus_yuksekligi)
                continue

            mekansal_cozunurluk = (camera_sensor_genislik * ucus_yuksekligi * 100) / (camera_focal_lenght * goruntu_piksel_genisligi)
            if mekansal_cozunurluk <= 0:
                print("Mekansal cozum gecersiz, atlaniyor:", mekansal_cozunurluk)
                continue
            olcek_scale_test = (mekansal_cozunurluk / 29.85)
                
                
            print("kamera model= ",kamera_model)
            print("focal lenght = ",FocalLength)
            print("altitude = : ",altitude)
            print("rakÃ„Â±m =: ",rakim)
            print("ucus_yuksekligi = :",ucus_yuksekligi)
            print("yaw = :",yaw)
            
            
    
            
            #################################################################################################
            
            # 7.1) AnlÃ„Â±k gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼yÃƒÂ¼ oku ve yaw/ÃƒÂ¶lÃƒÂ§ek ile dÃƒÂ¶ndÃƒÂ¼rmeye hazÃ„Â±rla
            # Reading the image
            image = cv2.imread(anlik_goruntu,0)
            image_color = cv2.imread(anlik_goruntu, cv2.IMREAD_COLOR)
            if image is None or image_color is None:
                print("Anlik goruntu okunamadi, atlaniyor:", anlik_goruntu)
                continue
            
            # dim=(1000,750)
            
            # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            
            # dividing height and width by 2 to get the center of the image
            height, width = image.shape[:2]
            # #get the center coordinates of the image to create the 2D rotation matrix
            #center = (int(width/2), int(height/2))
            
            # #using cv2.getRotationMatrix2D() to get the rotation matrix
            # #scale parametresi ile gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼nÃƒÂ¼n spartial ÃƒÂ§ÃƒÂ¶zÃƒÂ¼nÃƒÂ¼rlÃƒÂ¼Ã„Å¸ÃƒÂ¼ 60 cm'ye ayarlanÃ„Â±r
            # #angle ile gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼nÃƒÂ¼n yav deÃ„Å¸erinin tam tersine rotate edilir ve gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼ kuzeye dÃƒÂ¶ndÃƒÂ¼rÃƒÂ¼lÃƒÂ¼r.
            # rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=(-1*yaw), scale=olcek_scale)
            
            
            # #rotate the image using cv2.warpAffine
            # rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height), borderValue=(255,255,255))
            
            
            angle=-yaw
            rimage = rotate_image(image, angle)
            rimage_color = rotate_image(image_color, angle)
            
            #cv2.imwrite("rotate_edilmis.jpg", rimage)
           
            
            t = largest_rotated_rect(width, height, math.radians(angle))
            
            #cv2.imwrite("en_buyuk_ic_dortgen.jpg", t)
       
                
            #cv2.imshow("rotated",t)
            
            
            cr_image = crop_around_center(rimage,int(t[0]), int(t[1]))
            cr_image_color = crop_around_center(rimage_color,int(t[0]), int(t[1]))
            
            #cv2.imwrite("crop_edilmis.jpg", cr_image)
            
            
            #cv2.imshow("rotated",cr_image)
            
            height,width= (cr_image.shape[0],cr_image.shape[1])
            
            # ÃƒÅ“ÃƒÂ§ farklÃ„Â± ÃƒÂ¶lÃƒÂ§ek kullan: merkez, sol-ÃƒÂ¼st ve saÃ„Å¸-alt rakÃ„Â±ma gÃƒÂ¶re dÃƒÂ¼zelt
            if abs(rakim) < 1e-9:
                print("Rakim sifira cok yakin, atlaniyor")
                continue

            base_w = max(1, int(width * olcek_scale_test))
            base_h = max(1, int(height * olcek_scale_test))
            rotated_image = cuda_resize_if_available(cr_image, (base_w, base_h), interpolation=cv2.INTER_NEAREST)
            rotated_image_color = cuda_resize_if_available(cr_image_color, (base_w, base_h), interpolation=cv2.INTER_NEAREST)

            olcek_scale_sol_ust = olcek_scale_test * (rakim_sol_ust / rakim)
            olcek_scale_sag_alt = olcek_scale_test * (rakim_sag_alt / rakim)
            
            #olcek_scale_sol_ust=olcek_scale_test*(rakim/rakim_sol_ust)
            #olcek_scale_sag_alt=olcek_scale_test*(rakim/rakim_sag_alt)
            
            su_w = max(1, int(width * olcek_scale_sol_ust))
            su_h = max(1, int(height * olcek_scale_sol_ust))
            sa_w = max(1, int(width * olcek_scale_sag_alt))
            sa_h = max(1, int(height * olcek_scale_sag_alt))
            rotated_image_sol_ust = cuda_resize_if_available(cr_image, (su_w, su_h), interpolation=cv2.INTER_NEAREST)
            rotated_image_sag_alt = cuda_resize_if_available(cr_image, (sa_w, sa_h), interpolation=cv2.INTER_NEAREST)
            
            
            #ÃƒÂ§ÃƒÂ¶zÃƒÂ¼nÃƒÂ¼rlÃƒÂ¼Ã„Å¸ÃƒÂ¼ 30 cm'ye ayarlanmÃ„Â±Ã…Å¸ gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼nÃƒÂ¼n orta noktasÃ„Â± bulnur
            height, width = rotated_image.shape[:2]
            # get the center coordinates of the image to create the 2D rotation matrix
            center = (int(width/2), int(height/2))
            
            fark=np.minimum(center[0],center[1])-272    # 544'lÃ„Â±k frame'in elde edilen dikdÃƒÂ¶rtgenin dÃ„Â±Ã…Å¸Ã„Â±na taÃ…Å¸mamasÃ„Â± iÃƒÂ§in yazÃ„Â±ldÃ„Â± 
            if fark>200:
                fark=200
            elif fark<0:
                print("merkezi dÃ„Â±Ã…Å¸arÃ„Â±da")
                continue
           
            
            y1 = center[1]-PATCH_HALF-fark; y2 = center[1]+PATCH_HALF-fark
            x1 = center[0]-PATCH_HALF-fark; x2 = center[0]+PATCH_HALF-fark
            if not is_valid_slice(rotated_image_sol_ust, x1, y1, x2, y2):
                print("rotated_part1 sÃ„Â±nÃ„Â±r dÃ„Â±Ã…Å¸Ã„Â±nda")
                continue
            rotated_part1 = rotated_image_sol_ust[y1:y2, x1:x2]
            y1 = center[1]-PATCH_HALF; y2 = center[1]+PATCH_HALF
            x1 = center[0]-PATCH_HALF; x2 = center[0]+PATCH_HALF
            if not is_valid_slice(rotated_image, x1, y1, x2, y2):
                print("rotated_part2 sÃ„Â±nÃ„Â±r dÃ„Â±Ã…Å¸Ã„Â±nda")
                continue
            rotated_part2 = rotated_image[y1:y2, x1:x2]
            rotated_part2_color = rotated_image_color[y1:y2, x1:x2]
            y1 = center[1]-PATCH_HALF+fark; y2 = center[1]+PATCH_HALF+fark
            x1 = center[0]-PATCH_HALF+fark; x2 = center[0]+PATCH_HALF+fark
            if not is_valid_slice(rotated_image_sag_alt, x1, y1, x2, y2):
                print("rotated_part3 sÃ„Â±nÃ„Â±r dÃ„Â±Ã…Å¸Ã„Â±nda")
                continue
            rotated_part3 = rotated_image_sag_alt[y1:y2, x1:x2]
            
            
            
            # cv2.imshow('Original image', image)
            # cv2.imshow('Rotated image', rotated_image)
            if DEBUG:
                cv2.imshow('Rotated part', rotated_part2)
                _ = cv2.waitKey(1) 
            
            
            # 7.2) Template listesi (3 ÃƒÂ¶lÃƒÂ§ek)
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
            # 7.3) Model giriÃ…Å¸ ÃƒÂ¶n iÃ…Å¸lemleri (resize/equalize/normalize)
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
            # Show center crop and its neural output side-by-side
            try:
                vis_crop = _compose_top_bottom(
                    rotated_part2_color,
                    template[1],
                    top_title="Crop",
                    bottom_title="Model",
                    target_width=900,
                    apply_colormap_bottom=False,
                    caption_height=80,
                    gap=10,
                )
                if vis_crop is not None:
                    _show_image_fit("Crop vs Model", vis_crop, max_frac=0.75)
            except Exception:
                pass

            # Show: AnlÃ„Â±k (orijinal) ve Ã„Â°Ã…Å¸lenmiÃ…Å¸ (model ÃƒÂ§Ã„Â±ktÃ„Â±sÃ„Â±) yan yana
            try:
                vis_pair = _compose_side_by_side(
                    image,
                    template[1],
                    left_title="AnlÃ„Â±k",
                    right_title="Ã„Â°Ã…Å¸lenmiÃ…Å¸ (Model)",
                    target_height=900,
                    apply_colormap_right=True,
                )
                if False and vis_pair is not None:
                    # _show_image_fit disabled: using Crop vs Model window instead
                    # _show_image_fit("Anlik-Islenmis (GENIS)", vis_pair, max_frac=0.98)
                    cv2.namedWindow("AnlÃ„Â±k vs Ã„Â°Ã…Å¸lenmiÃ…Å¸", cv2.WINDOW_NORMAL)
                    cv2.imshow("AnlÃ„Â±k vs Ã„Â°Ã…Å¸lenmiÃ…Å¸", vis_pair)
            except Exception:
                pass
                
            
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
            
            
            #paralel programlama ile aynÃ„Â± anda 3 templatematching yapÃ„Â±lÃ„Â±r
            inputs=[(cerceve,template[0]),(cerceve,template[1]),(cerceve,template[2])]
            # Template matching (sequential to avoid IPC overhead)
            res1, res2, res3 = match_three(cerceve, [template[0], template[1], template[2]])
            # Not: CUDA varsa tek seferde gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼ yÃƒÂ¼klenip ÃƒÂ¼ÃƒÂ§ eÃ…Å¸leÃ…Å¸me GPUÃ¢â‚¬â„¢da yapÃ„Â±lÃ„Â±r; aksi halde CPU.
            #methods =['cv2.TM_CCOEFF']
            #for meth in methods:
                #method  = eval(meth)    #stringleri fonksiyona ÃƒÂ§eviren fonksiyona
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
                    
            # max_loc = (x, y) ROI icindeki koordinattir; global konum icin ROI ofseti eklenir.
            top_left1 = (max_loc1[0] + ust, max_loc1[1] + sol)
            top_left2 = (max_loc2[0] + ust, max_loc2[1] + sol)
            top_left3 = (max_loc3[0] + ust, max_loc3[1] + sol)
            
            
         
            bottom_right1 = (top_left1[0] + w,top_left1[1] + h)
            bottom_right2 = (top_left2[0] + w,top_left2[1] + h)
            bottom_right3 = (top_left3[0] + w,top_left3[1] + h)
            
                #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                
            # global_x = max_loc2[0]+int(w/2)
            # global_y = max_loc2[1]+int(h/2)
                 
            
                 
            # ÃƒÅ“ÃƒÂ§ aday dikdÃƒÂ¶rtgenin (x,y,w,h) biÃƒÂ§iminde paketlenmesi
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
                
                 
                 #konum bulmak iÃƒÂ§in kesiÃ…Å¸imler ve kesiÃ…Å¸im karelerinin koordinatlarÃ„Â± bulunuyor
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
                print("kesiÃ…Å¸im yok")
                kare=(0,0,0,0)
                kare=b
                cerceve_boyutu+=500
                yanlis_negatif_kontrol+=1
                
            
            
            
            
            # KesiÃ…Å¸im merkezinin koordinatÃ„Â± (piksel cinsinden)
            konum_y=kare[0]+int(kare[2]/2)
            konum_x=kare[1]+int(kare[3]/2)
            if konum_y < 0:
                konum_y = 0
            if konum_x < 0:
                konum_x = 0
                
            if konum_y>img.shape[1]:
                konum_y=img.shape[1]-1
            if konum_x>img.shape[0]:
                konum_x=img.shape[0]-1                
                     
            
            
            konum=(konum_y,konum_x)
            
           
            
            
            
            
            #konum = (kare[0]+int(kare[2]/2),kare[1]+int(kare[3]/2))
            
            """
                gps_longtidye ce gps_latitde deÃ„Å¸iÃ…Å¸kenleri anlÃ„Â±k gÃƒÂ¶rÃƒÂ¼ntÃƒÂ¼nÃƒÂ¼n korrdinatlarÃ„Â±nÃ„Â± verir
                koordinatlar[1][konum[0]][konum[1]] ise modelin tahmin ettiÃ„Å¸i konumun koordinatlarÃ„Â±nÃ„Â± verir
                ve aralarÃ„Â±ndaki uzaklÃ„Â±k hesaplanÃ„Â±r.    
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
            
            
            
            # BaÃ…Å¸arÃ„Â± eÃ…Å¸iÃ„Å¸i: 70 metre (0.07 km). Duruma gÃƒÂ¶re TP/FP/TN/FN sayaÃƒÂ§larÃ„Â± gÃƒÂ¼ncellenir.
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
               
                
               
            if uzaklik > 0.3 and benchmark == False:
                # Ilk karede geri donus noktasi (0,0) olmasin; EXIF konumuna don.
                if i == 0:
                    konum = (knm[1], knm[0])
                else:
                    konum = konum_once
               
          
                
            
                
            # dosyaya_yaz(sonuclar,dogru_tahmin,yanlis_tahmin)  # DÃƒÂ¶ngÃƒÂ¼ sonunda bir defa yazÃ„Â±lacak
                
            centerOfCircle=konum    
                    
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            if benchmark==True:
                cerceve_boyutu=cerceve_boyutu_deger
                konum=(knm[1],knm[0])
                
            
            if runtime_ui_state.get("inner_frame", True):
                cv2.rectangle(
                    img,
                    (-int(cerceve_boyutu/2)+konum[0], -int(cerceve_boyutu/2)+konum[1]),
                    (+int(cerceve_boyutu/2)+konum[0], +int(cerceve_boyutu/2)+konum[1]),
                    (0, 0, 0),
                    25
                )

            if runtime_ui_state.get("tm_boxes", True):
                cv2.rectangle(img, top_left1, bottom_right1, (0, 0, 255), 25)
                cv2.rectangle(img, top_left2, bottom_right2, (0, 255, 0), 25)
                cv2.rectangle(img, top_left3, bottom_right3, (255, 0, 0), 25)
            radius=10

            pred_pt = (int(centerOfCircle[0]), int(centerOfCircle[1]))
            real_pt = (int(knm[1]), int(knm[0]))

            # Noktalari sadece gecerli piksel araliginda kaydet.
            if 0 <= pred_pt[0] < img.shape[1] and 0 <= pred_pt[1] < img.shape[0]:
                traj_pred_points.append(pred_pt)
            if 0 <= real_pt[0] < img.shape[1] and 0 <= real_pt[1] < img.shape[0]:
                traj_real_points.append(real_pt)

            if TRAJECTORY_MAX_POINTS > 0:
                if len(traj_pred_points) > TRAJECTORY_MAX_POINTS:
                    traj_pred_points = traj_pred_points[-TRAJECTORY_MAX_POINTS:]
                if len(traj_real_points) > TRAJECTORY_MAX_POINTS:
                    traj_real_points = traj_real_points[-TRAJECTORY_MAX_POINTS:]

            if runtime_ui_state.get("trajectory", False):
                # Trajektori cizgileri: tahmin=sari, gercek=yesil
                if len(traj_pred_points) >= 2:
                    pts_pred = np.array(traj_pred_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        img, [pts_pred], isClosed=False, color=(0, 255, 255),
                        thickness=max(1, TRAJECTORY_LINE_THICKNESS), lineType=cv2.LINE_AA
                    )
                if len(traj_real_points) >= 2:
                    pts_real = np.array(traj_real_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        img, [pts_real], isClosed=False, color=(0, 255, 0),
                        thickness=max(1, TRAJECTORY_LINE_THICKNESS), lineType=cv2.LINE_AA
                    )

                # Her adim nokta izi
                if TRAJECTORY_DRAW_POINTS:
                    rp = max(1, TRAJECTORY_POINT_RADIUS)
                    for p in traj_pred_points:
                        cv2.circle(img, p, rp, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                    for p in traj_real_points:
                        cv2.circle(img, p, rp, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            cv2.circle(img, centerOfCircle, radius, (0,255,255), 25)   #tahmini konumu veren nokta
            cv2.circle(img,(knm[1],knm[0]),radius,(0,255,0), 25)                   #gerÃƒÂ§ek konumu gÃƒÂ¶steren nokta
                    #plt.figure()
                    
                
                    # plt.imshow(img)
                    # plt.title("Tespit edilen SonuÃƒÂ§"), plt.axis("on")
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
                
            
            # UÃƒÂ§ak simgesini (gerÃƒÂ§ek konum ve heading ile) ÃƒÂ§iz
            try:
                draw_plane_icon_v2(img, (knm[1], knm[0]), yaw, size_px=220, color=(255,0,255), outline=(0,0,0), outline_thickness=10)
            except Exception:
                pass

            res = img[ressol:ressag,resust:resalt] 
                
                
                
            window_name = 'Image'

            # 7.4) HUD: baÃ…Å¸lÃ„Â±k (yaw), uÃƒÂ§uÃ…Å¸ yÃƒÂ¼ksekliÃ„Å¸i ve hatayÃ„Â± gÃƒÂ¶ster; ÃƒÂ¶lÃƒÂ§ek ÃƒÂ§ubuÃ„Å¸u ve hedef iÃ…Å¸aretleri ÃƒÂ§iz
            hud_lines = [
                f"HDG: {yaw:.1f} deg",
                f"ALT: {int(ucus_yuksekligi)} m",
                f"ERR: {int(uzaklik*1000)} m",
            ]
            # HUD panelini sol alt koseye yerlestir (butonlarla cakismasin)
            _hud_font_scale = 6
            _hud_thickness = 20
            _hud_padding = 25
            try:
                _hud_sizes = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX,
                              _hud_font_scale, _hud_thickness)[0] for s in hud_lines]
                _hud_max_h = max(h for (_, h) in _hud_sizes)
                _hud_line_gap = int(_hud_max_h * 1.6)
                _hud_panel_h = _hud_line_gap * len(hud_lines) + _hud_padding
                _hud_y = max(150, res.shape[0] - _hud_panel_h - 40)
            except Exception:
                _hud_y = max(150, res.shape[0] - 600)
            draw_info_panel(
                res,
                hud_lines,
                top_left=(25, _hud_y),
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=_hud_font_scale,
                thickness=_hud_thickness,
                alpha=0.55,
                padding=_hud_padding,
                corner_radius=18,
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




                
                
            if UI_BUTTONS_ENABLED:
                try:
                    # Mouse callback icin goruntu boyutunu guncelle
                    runtime_ui_ctx["img_size"] = (res.shape[1], res.shape[0])
                    _draw_runtime_buttons(
                        res,
                        runtime_ui_state,
                        runtime_ui_buttons,
                        font_scale=UI_BUTTON_FONT_SCALE,
                        thickness=UI_BUTTON_THICKNESS,
                        ui_scale=UI_BUTTON_SCALE,
                        display_size=(UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT),
                    )
                except Exception:
                    pass

            cv2.namedWindow("konum", cv2.WINDOW_NORMAL)  
            cv2.resizeWindow("konum", UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT)
            if UI_BUTTONS_ENABLED and (not runtime_ui_cb_set):
                try:
                    cv2.setMouseCallback("konum", _runtime_buttons_mouse_cb, runtime_ui_ctx)
                    runtime_ui_cb_set = True
                except Exception:
                    pass
            cv2.imshow("konum", res)
            key = cv2.waitKey(1) & 0xFF
            if UI_BUTTONS_ENABLED:
                if key in (ord('t'), ord('T')):
                    runtime_ui_state["trajectory"] = not bool(runtime_ui_state.get("trajectory", False))
                elif key in (ord('i'), ord('I')):
                    runtime_ui_state["inner_frame"] = not bool(runtime_ui_state.get("inner_frame", True))
                elif key in (ord('r'), ord('R')):
                    runtime_ui_state["tm_boxes"] = not bool(runtime_ui_state.get("tm_boxes", True))
                elif key in (ord('h'), ord('H')):
                    runtime_ui_state["_panel_collapsed"] = not bool(runtime_ui_state.get("_panel_collapsed", False))
            
                # cv2.rectangle(img, top_left, bottom_right,(255,0,0),35)
                # plt.figure()
                # plt.subplot(121), plt.imshow(res, cmap = "gray")
                # plt.title("EÃ…Å¸leÃ…Å¸en SonuÃƒÂ§"), plt.axis("on")
                # plt.subplot(122), plt.imshow(img)
                # plt.title("Tespit edilen SonuÃƒÂ§"), plt.axis("on")
                # plt.suptitle(meth)
                # img = cv2.imread(harita,0)
        
        
            bitis_zamani = time.time()
            calisma_suresi = bitis_zamani - baslangic_zamani
            #print("Kodun calisma suresi___________________________________________:", calisma_suresi, "saniye")
            
            print("Kodun calisma suresi___________________________________________:", "{:.2f}".format(calisma_suresi), "saniye")
            

            print((i+1),"/",(len(anlik_yol_list)),"     dogru_tahmin: ,"+str(dogru_tahmin)+",  yanlis_tahmin: ,"+str(yanlis_tahmin) +",  dogru pozitif: "+str(dogru_pozitif)+",  yanlÃ„Â±Ã…Å¸ pozitif: "+str(yanlis_pozitif)+",  dogru negatif: "+str(dogru_negatif)+",  yanlÃ„Â±Ã…Å¸ negatif: "+str(yanlis_negatif)+"\n")
         
        
        # 8) DÃƒÂ¶ngÃƒÂ¼ sonu: kaynaklarÃ„Â± serbest bÃ„Â±rak, hata metriklerini hesapla ve ÃƒÂ§Ã„Â±ktÃ„Â± dosyalarÃ„Â±na yaz
        # Close map dataset for this loop to free resources
        try:
            map_ds.close()
        except Exception:
            pass
        uzaklik_hatalari = np.array(uzaklik_hatalari)

        if uzaklik_hatalari.size > 0:
            rmse_degeri = rmse(uzaklik_hatalari)
            mae_degeri = mae(uzaklik_hatalari)
            standart_sapma_degeri = standart_sapma(uzaklik_hatalari)
        else:
            rmse_degeri = float('nan')
            mae_degeri = float('nan')
            standart_sapma_degeri = float('nan')
        
        
        # Yeni verilen deÃ„Å¸erler iÃƒÂ§in tekrar hesaplama yapÃ„Â±lÃ„Â±yor


        
        # Hassasiyet ve Geri Cagirma hesaplamalari
        def _safe_div(num, den, default=0.0):
            return (num / den) if den else default

        hassasiyet_yeni = _safe_div(dogru_pozitif, (dogru_pozitif + yanlis_pozitif))
        geri_cagirma_yeni = _safe_div(dogru_pozitif, (dogru_pozitif + yanlis_negatif))
        
        # F Skoru hesaplama
        f_skoru = _safe_div(2 * (hassasiyet_yeni * geri_cagirma_yeni), (hassasiyet_yeni + geri_cagirma_yeni))
        

        
        sonuclar_dosya = open("modele_gore_sonuclar.txt", "a+")

       # sonuclar = np.vstack((sonuclar,dogru_tahmin, yanlis_tahmin)).T
       # print(sonuclar)
        sonuclar_=anlik_yol[-20:]+" "+str(model_list[k])+",  dogru_tahmin: ,"+str(dogru_tahmin)+",  yanlis_tahmin: ,"+str(yanlis_tahmin) + ",  RMSE DeÃ„Å¸eri: ,"+str(rmse_degeri)+ ",  MAE DeÃ„Å¸eri: ,"+str(mae_degeri)+",  standart sapma DeÃ„Å¸eri: ,"+str(standart_sapma_degeri)+",  dogru pozitif: ,"+str(dogru_pozitif)+",  yanlÃ„Â±Ã…Å¸ pozitif: ,"+str(yanlis_pozitif)+",  dogru negatif: ,"+str(dogru_negatif)+",  yanlÃ„Â±Ã…Å¸ negatif: ,"+str(yanlis_negatif)+", f skoru: ,"+str(f_skoru)+"\n"
     
        sonuclar_dosya.write(sonuclar_)
        sonuclar_dosya.close()
      
            
        print("dogru tahmin = ",dogru_tahmin)
        print("yanlÃ„Â±Ã…Å¸ tahmin = ",yanlis_tahmin)
        print("yanlÃ„Â±Ã…Å¸ pozitif = ",yanlis_pozitif)
        yuzde = _safe_div(dogru_tahmin, (dogru_tahmin + yanlis_tahmin))
        yuzde=yuzde*100
        print("doÃ„Å¸ruluk yÃƒÂ¼zdesi: {:.2f}".format(yuzde))
        
        # DÃƒÂ¶ngÃƒÂ¼ sonu: sonuÃƒÂ§larÃ„Â± bir defa yaz
        try:
            dosyaya_yaz(sonuclar, dogru_tahmin, yanlis_tahmin)
        except Exception as _e:
            print("sonuclar dosyaya yazÃ„Â±lÃ„Â±rken hata:", _e)

        if bool(RUN_CFG.get("WAIT_PER_MODEL", False)):
            input("pause")

    try:
        dem_ds.close()
    except Exception:
        pass
    try:
        del dataset
    except Exception:
        pass
    if bool(RUN_CFG.get("WAIT_ON_EXIT", False)):
        input("pause")


