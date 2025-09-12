import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
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
import os
import piexif
import csv
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

from rasterio.warp import transform
from PIL import Image
from PIL.ExifTags import TAGS

from pyproj import Transformer
warnings.filterwarnings("ignore")


# import rasterio as rio
# from rasterio.warp import transform 
# import matplotlib.pyplot as plt
# import pandas as pd
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img



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
    
    
    
from pyproj import Proj, transform

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
    
    # Create a Proj object for WGS84
    wgs84 = Proj(proj='latlong', datum='WGS84')
    
    # Create a Proj object for the UTM zone
    utm = Proj(proj='utm', zone=zone_number, datum='WGS84')
    
    # Perform the coordinate transformation
    easting, northing = transform(wgs84, utm, longitude, latitude)
    
    return easting, northing, zone_number, hemisphere


from math import sin, cos, sqrt, atan2, radians

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

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

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
benchmark=True

# Global debug and sizing constants
DEBUG = False
PATCH_SIZE = 544
PATCH_HALF = PATCH_SIZE // 2
PRED_BORDER = 16

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
    methods =['cv2.TM_CCOEFF_NORMED']
    method  = eval(methods[0])
    
    res= cv2.matchTemplate(img, template, method, None)
    return res
    


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
    
    #haritalar klasöründeki ilk görüntüde DEM verileri vardır. ikinci görüntü ise normal rgb görüntüdür.
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
    from pyproj import Proj, transform, Transformer
 
    #fname = 'urgup_gmap_georef.tif'
    fname = harita_yol+harita_yol_list[0]
    # Read raster
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        print(p1)
        A = r.read()  # pixel values
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))
    
    def koordinat_bul(row,col):
        # Get affine transform for pixel centres
        T1 = T0 * Affine.translation(0.5, 0.5)
        # Function to convert pixel row/column index (from 0) to easting/northing at centre
        rc2en = lambda r, c: (c, r) * T1
        
        # All eastings and northings (there is probably a faster way to do this)
        eastings, northings = np.vectorize(rc2en, otypes=[float, float])([row], [col])
        
        
        # Project all longitudes, latitudes
        p2 = Proj(proj='latlong',datum='WGS84')
        
        #p2 = Proj(proj='utm', zone=36, datum='WGS84')  # UTM Zone 33T  #silinecek
        
        longs, lats = transform(p1, p2, eastings, northings)
       
        
        return (longs,lats)
    
    
    #%%
    
    # pickle_in = open("koordinatlar.pickle","rb")
    # koordinatlar = pickle.load(pickle_in)
    
    
    # print(koordinatlar[0][10][10])
    # print(koordinatlar[1][10][10])
    ###############################################################################
    
    #DEM verileri aktarılır
    
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
          
        t_img = cv2.imread(ana_harita,0)  #haritalar klasöründeki ikinci görüntüyü okur
        print(t_img.shape)

        # Open main map once per loop and reuse for pixel lookups
        map_ds = rio.open(ana_harita)
        ll_to_map = Transformer.from_crs("EPSG:4326", map_ds.crs, always_xy=True)
          
        kenarx=int(t_img.shape[0]/512)
        
        #parcalar klasöründeki anlık görüntüleri getirir
        anlik_yol = os.getenv("ANLIK_YOL", os.path.join(dirname, 'parcalar'))
        
        #anlik_yol="parcalar/"
        
        anlik_yol_list=os.listdir(anlik_yol)
        
        #anlik_goruntu=anlik_yol+anlik_yol_list[0]
        
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
            knm = piksel_bul_fast(map_ds, ll_to_map, gps_longitude, gps_latitude)
            

            
            
                       
                
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
            
            rotated_image=cv2.resize(cr_image, (int(width*olcek_scale_test),int(height*olcek_scale_test)),interpolation=cv2.INTER_NEAREST ) 
            
            olcek_scale_sol_ust=olcek_scale_test*(rakim_sol_ust/rakim)
            olcek_scale_sag_alt=olcek_scale_test*(rakim_sag_alt/rakim)
            
            #olcek_scale_sol_ust=olcek_scale_test*(rakim/rakim_sol_ust)
            #olcek_scale_sag_alt=olcek_scale_test*(rakim/rakim_sag_alt)
            
            rotated_image_sol_ust=cv2.resize(cr_image, (int(width*olcek_scale_sol_ust),int(height*olcek_scale_sol_ust)),interpolation=cv2.INTER_NEAREST ) 
            rotated_image_sag_alt=cv2.resize(cr_image, (int(width*olcek_scale_sag_alt),int(height*olcek_scale_sag_alt)),interpolation=cv2.INTER_NEAREST ) 
            
            
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
           
            
            rotated_part1 = rotated_image_sol_ust[center[1]-PATCH_HALF-fark:center[1]+PATCH_HALF-fark,center[0]-PATCH_HALF-fark:center[0]+PATCH_HALF-fark]
            rotated_part2 = rotated_image[center[1]-PATCH_HALF:center[1]+PATCH_HALF,center[0]-PATCH_HALF:center[0]+PATCH_HALF]
            rotated_part3 = rotated_image_sag_alt[center[1]-PATCH_HALF+fark:center[1]+PATCH_HALF+fark,center[0]-PATCH_HALF+fark:center[0]+PATCH_HALF+fark]
            
            
            
            # cv2.imshow('Original image', image)
            # cv2.imshow('Rotated image', rotated_image)
            if DEBUG:
                cv2.imshow('Rotated part', rotated_part2)
                _ = cv2.waitKey(1) 
            
            
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
            res1 = match(cerceve, template[0])
            res2 = match(cerceve, template[1])
            res3 = match(cerceve, template[2])
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
            koordinatlar=koordinat_bul(konum[1],konum[0])
            
            
            lat_tahmin = koordinatlar[1][0]
            long_tahmin = koordinatlar[0][0]
            
                
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
                
            
            res = img[ressol:ressag,resust:resalt] 
                
                
                
            window_name = 'Image'
  
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
              
            # org
            org = (25, 150)
              
            # fontScale
            fontScale = 6
               
            # Blue color in BGR
            color = (0, 0, 255)
              
            # Line thickness of 2 px
            thickness = 25
               
            text="hdg: "+str(yaw)+"' "
            # Using cv2.putText() method
            cv2.putText(res, text, org, font, 
                               fontScale, color, thickness, cv2.LINE_AA)
            
            text="alt: "+str(int(ucus_yuksekligi))+" metre"
            org = (25, 325)
            cv2.putText(res, text, org, font, 
                               fontScale, color, thickness, cv2.LINE_AA)
                
                
                
                
                
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

        input("pause")
