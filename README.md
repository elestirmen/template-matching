# UAV Görüntü Konumlandırma - Template Matching ile Derin Öğrenme Destekli Jeolokalizasyon

Bu proje, İHA (İnsansız Hava Aracı) / drone görüntülerini **referans ortofoto harita** üzerinde otomatik olarak konumlandıran bir bilgisayarlı görü sistemidir. Derin öğrenme tabanlı öznitelik çıkarımı ile çok ölçekli template matching yöntemini birleştirerek, GPS'ten bağımsız veya GPS doğrulamalı konum tahmini gerçekleştirir.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Çalışma Akışı (Pipeline)](#çalışma-akışı-pipeline)
- [Klasör Yapısı](#klasör-yapısı)
- [Yapılandırma (RUN\_CFG)](#yapılandırma-run_cfg)
- [Kurulum](#kurulum)
- [Bağımlılıklar](#bağımlılıklar)
- [Kullanım](#kullanım)
  - [Runtime Kontrolleri](#runtime-kontrolleri)
- [Teknik Detaylar](#teknik-detaylar)
  - [EXIF Verisi Okuma](#exif-verisi-okuma)
  - [DEM Tabanlı Çok Ölçekli Yaklaşım](#dem-tabanlı-çok-ölçekli-yaklaşım)
  - [Görüntü Ön İşleme](#görüntü-ön-işleme)
  - [Keras Model Çıkarımı](#keras-model-çıkarımı)
  - [Template Matching](#template-matching)
  - [Konum Belirleme (Kesişim Yöntemi)](#konum-belirleme-kesişim-yöntemi)
  - [Piramit Arama (Pyramid Search)](#piramit-arama-pyramid-search)
  - [CUDA GPU Hızlandırma](#cuda-gpu-hızlandırma)
  - [Kalman Filtresi (Konum Takibi)](#kalman-filtresi-konum-takibi)
- [Koordinat Dönüşümleri](#koordinat-dönüşümleri)
- [Değerlendirme Metrikleri](#değerlendirme-metrikleri)
- [Görselleştirme](#görselleştirme)
- [Çıktı Dosyaları](#çıktı-dosyaları)
- [Kamera Desteği](#kamera-desteği)
- [Sınırlamalar ve Notlar](#sınırlamalar-ve-notlar)

---

## Genel Bakış

Sistem, bir drone'un çektiği anlık görüntüleri georeferanslı bir ortofoto harita üzerinde arayarak konumlandırır. Temel fikir şudur:

1. Drone görüntüsünün EXIF meta verisinden **yaw (başlık açısı)**, **GPS koordinatı**, **odak uzaklığı** ve **irtifa** bilgileri okunur.
2. Sayısal Yükseklik Modeli (DEM) kullanılarak **arazi rakımı** belirlenir ve uçuş yüksekliği hesaplanır.
3. Uçuş yüksekliği ve kamera parametreleri ile **Yer Örneklem Aralığı (GSD)** hesaplanır.
4. Görüntü yaw açısına göre kuzeye hizalanır ve GSD'ye göre ölçeklenir.
5. Üç farklı ölçekte (merkez, sol-üst, sağ-alt rakım değerlerine göre) **template patch**'leri üretilir.
6. Her patch bir **Keras derin öğrenme modeli** ile işlenerek öznitelik haritası çıkarılır.
7. Referans haritanın ilgili bölgesinde **template matching** (Normalized Cross-Correlation) yapılır.
8. Üç eşleşmenin kesişim alanından **nihai konum** belirlenir.
9. Tahmin edilen konum ile gerçek GPS konumu arasındaki hata hesaplanır.

---

## Sistem Mimarisi

```
+---------------------------------------------------------------------+
|                         DRONE GORUNTULERI                            |
|                        (parcalar/ klasoru)                           |
+--------------------------------+------------------------------------+
                                 |
                    +------------v------------+
                    |    EXIF Okuyucu         |
                    |  (yaw, GPS, alt,        |
                    |   focal length)         |
                    +------------+------------+
                                 |
              +------------------+------------------+
              |                  |                  |
    +---------v--------+ +------v------+ +---------v---------+
    |  DEM Sorgusu     | | GSD Hesabi  | | Yaw Duzeltmesi    |
    |  (Rakim Bulma)   | | (cm/piksel) | | (Kuzeye Hizala)   |
    +---------+--------+ +------+------+ +---------+---------+
              |                  |                  |
              +------------------+------------------+
                                 |
                    +------------v------------+
                    |  Cok Olcekli Patch      |
                    |   Uretimi (3 adet)      |
                    |  Sol-ust / Merkez /     |
                    |     Sag-alt rakim       |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |   Keras Modeli          |
                    |  (Oznitelik Cikarim)    |
                    |   Batch predict         |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |  Template Matching      |
                    |  (TM_CCOEFF_NORMED)     |
                    |  CPU veya CUDA GPU      |
                    |  Piramit arama opt.     |
                    +------------+------------+
                                 |
                    +------------v------------+
                    |  Kesisim Analizi        |
                    |  (3 sonucun birlesimi)  |
                    +------------+------------+
                                 |
              +------------------+------------------+
              |                  |                  |
    +---------v--------+ +------v------+ +---------v---------+
    | Konum Tahmini    | | Hata Hesabi | |  Gorsellestirme   |
    | (Lat/Lon)        | | (Haversine) | |  (OpenCV HUD)     |
    +------------------+ +-------------+ +-------------------+
```

---

## Calisma Akisi (Pipeline)

Betik calistirildiginda asagidaki adimlar sirayla gerceklestirilir:

### 1. Baslatma ve Kaynak Yukleme
- `RUN_CFG` yapilandirma sozlugunden tum parametreler okunur.
- Harita dosyalari (`haritalar/`), model dosyalari (`model/`) ve anlik goruntuler (`parcalar/`) listelenir.
- DEM (Sayisal Yukseklik Modeli) raster dosyasi GDAL ile acilir.
- Referans haritanin CRS (Koordinat Referans Sistemi) bilgisi okunur ve `piksel -> koordinat` donusturucusu hazirlanir.

### 2. Model ve Harita Dongusu
Her harita-model cifti icin:
- Keras modeli (`load_model`) yuklenir.
- Referans harita gri tonlu olarak OpenCV ile okunur.
- Rasterio ile haritanin CRS/transform bilgileri alinir.

### 3. Anlik Goruntu Isleme Dongusu
Her drone goruntusu icin:

#### 3.1 EXIF Okuma
- PIL/Pillow kullanilarak EXIF verisi parse edilir.
- Yaw acisi (MakerNote -> FlightDegree), GPS koordinatlari, irtifa ve odak uzakligi cikarilir.

#### 3.2 DEM Sorgulama ve GSD Hesabi
- Drone'un GPS konumuna karsilik gelen DEM pikselinden **arazi rakimi** okunur.
- Sol-ust ve sag-alt koselerdeki rakim degerleri de ayrica alinir.
- `Ucus Yuksekligi = GPS Irtifasi - Arazi Rakimi + Duzeltme`
- `GSD (cm/px) = (Sensor Genisligi x Ucus Yuksekligi x 100) / (Odak Uzakligi x Goruntu Genisligi)`

#### 3.3 Goruntu On Isleme
- Goruntu **yaw acisinin tersi** kadar dondurulerek kuzeye hizalanir.
- Dondurulen goruntuden **en buyuk ic dortgen** kesilir (siyah koseler kaldirilir).
- GSD oraniyla olceklenerek referans harita cozunurlugune getirilir.
- Uc farkli olcekle (merkez, sol-ust, sag-alt rakima gore) 544x544 piksellik patch'ler kesilir.

#### 3.4 Derin Ogrenme Model Cikarimi
- 3 patch, tek bir batch olarak modele verilir.
- Histogrami esitleme ve [-1, 1] normalizasyonu uygulanir.
- Model ciktisi 0-255 araligina donusturulur ve kenarlik pikselleri (`PRED_BORDER`) kirpilir.

#### 3.5 Template Matching
- Referans haritanin arama cercevesi bolumunde `cv2.TM_CCOEFF_NORMED` ile eslestirme yapilir.
- CUDA GPU varsa `cv2.cuda.createTemplateMatching` ile hizlandirilmis eslestirme kullanilir.
- Istege bagli piramit arama: once dusuk cozunurlukte kaba arama, ardindan bulunan bolgede ince arama.
- 3 template icin eszamanli eslestirme (`ThreadPoolExecutor`).

#### 3.6 Konum Belirleme
- 3 eslestirme sonucunun dortgenleri arasindaki **kesisim alani** hesaplanir.
- Kesisim merkezinin piksel koordinati -> cografi koordinat donusumu yapilir.
- Haversine formulu ile tahmin-gercek arasi mesafe hesaplanir.

#### 3.7 Adaptif Arama Cercevesi
- Normal modda: bir sonraki kare icin arama cercevesi, mevcut tahmine yakin bolgeye daraltilir.
- Eslestirme basarisizsa cerceve genisletilir.
- Benchmark modunda: her kare icin EXIF GPS merkezli sabit cerceve kullanilir.

### 4. Sonuc Raporlama
- RMSE, MAE, standart sapma hesaplanir.
- Precision (hassasiyet), Recall (geri cagirma) ve F-skoru hesaplanir.
- Sonuclar `sonuclar.csv`, `sonuclar.txt` ve `modele_gore_sonuclar.txt` dosyalarina yazilir.

---

## Klasor Yapisi

```
template matching/
|
+-- template_matching_parallel_processing_560_hizli_solust_sagalt_
|   koordinat_fonksiyonlar_icinde_cursor.py   # Ana betik
|
+-- haritalar/              # Georeferansli ortofoto harita dosyalari (.tif)
+-- model/                  # Keras derin ogrenme modelleri (.h5)
+-- parcalar/               # Islenecek drone/IHA goruntuleri
|
+-- anlik/                  # Anlik goruntu klasoru (alternatif)
+-- anlik_t/                # Anlik goruntu klasoru (alternatif)
+-- temp/                   # Gecici dosyalar
+-- haritalar_top/          # Ek harita dosyalari
+-- top_modeller/           # Ek model dosyalari
+-- arsiv/                  # Arsivlenmis dosyalar
|
+-- bern sehri template match/  # Bern sehri test verileri
|
+-- ana_harita_urgup_30_cm_utm_elevation.tif  # DEM dosyasi (varsayilan)
|
+-- sonuclar.csv            # Cikti: detayli sonuclar (CSV)
+-- sonuclar.txt            # Cikti: detayli sonuclar (metin)
+-- modele_gore_sonuclar.txt # Cikti: model bazli ozet metrikler
|
+-- README.md               # Bu dosya
+-- .gitignore
```

---

## Yapilandirma (RUN_CFG)

Tum calisma parametreleri dosyanin basindaki tek `RUN_CFG` sozlugunden yonetilir. Bu sozluk okunduktan sonra tip-guvenli sabitlere (bool/int/float) donusturulur.

| Parametre | Varsayilan | Aciklama |
|-----------|-----------|----------|
| `BENCHMARK` | `False` | `True` ise her kare EXIF GPS merkezi etrafinda sabit cerceve kullanir (adaptif takip kapali) |
| `DEBUG` | `False` | `True` ise ara goruntuler (patch, model ciktisi) ekranda gosterilir |
| `PATCH_SIZE` | `544` | Model giris boyutu (piksel) |
| `PRED_BORDER` | `16` | Model ciktisindan kirpilacak kenarlik (piksel) |
| `USE_PYRAMID` | `True` | Piramit (coarse-to-fine) arama etkinlestir |
| `COARSE_SCALE` | `0.5` | Piramit arama kaba olcek faktoru |
| `ROI_PAD_FACTOR` | `0.4` | Piramit arama ince arama bolgesi genisleme katsayisi |
| `CERCEVE_BOYUTU_NORMAL` | `2048` | Normal modda arama cercevesi boyutu (piksel) |
| `CERCEVE_BOYUTU_BENCHMARK` | `5000` | Benchmark modunda arama cercevesi boyutu (piksel) |
| `HARITA_DIR` | `"haritalar"` | Harita dosyalarinin bulundugu klasor |
| `MODEL_DIR` | `"model"` | Keras model dosyalarinin bulundugu klasor |
| `ANLIK_DIR` | `"parcalar"` | Drone goruntularinin bulundugu klasor |
| `DEM_PATH` | `"ana_harita_urgup_30_cm_utm_elevation.tif"` | DEM raster dosyasi yolu |
| `HARITA_DOSYALARI` | `[]` | Belirli harita dosyalari listesi (bos = klasordeki tumu) |
| `MODEL_DOSYALARI` | `[]` | Belirli model dosyalari listesi (bos = klasordeki tumu) |
| `SORT_INPUTS` | `False` | Girdi dosyalarini alfabetik sirala |
| `DEFAULT_FOCAL_LENGTH_MM` | `8.8` | EXIF'te yoksa varsayilan odak uzakligi (mm) |
| `DEFAULT_SENSOR_WIDTH_MM` | `13.2` | Bilinmeyen kamera icin varsayilan sensor genisligi (mm) |
| `USE_GPS_ALT_REF_SIGN` | `False` | GPS altitude referans isaretini uygula |
| `WAIT_PER_MODEL` | `False` | Her model sonrasi durakla |
| `WAIT_ON_EXIT` | `False` | Program sonunda durakla |
| `LOG_LEVEL` | `"WARNING"` | Opsiyonel logger seviyesi (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Varsayilan `WARNING` oldugundan normal calismada ek cikti uretmez |
| `LOG_TO_FILE` | `False` | `True` ise loglar ayrica `tm_run.log` dosyasina yazilir |
| `MAP_RES_CM_PER_PX` | `29.85` | Referans harita cozunurlugu (cm/piksel). Olcekleme ve hiz hesabinda kullanilir |
| `KENAR_SINIR_PX` | `272` | Harita kenarina bu kadar yakin konumlar atlanir (piksel) |

Kalman filtresi (konum takibi) ayarlari -- `USE_KALMAN=False` iken mevcut davranis birebir korunur:

| Parametre | Varsayilan | Aciklama |
|-----------|-----------|----------|
| `USE_KALMAN` | `False` | `True` ise konum tahmini Kalman ile filtrelenir (yalnizca `BENCHMARK=False`'te etkin) |
| `KALMAN_PROCESS_NOISE` | `50.0` | Surec gurultu std (px). Buyudukce olcume daha cabuk uyar (az yumusatma) |
| `KALMAN_MEASUREMENT_NOISE` | `8.0` | Olcum gurultu std (px). Kucuk -> iyi karelerde olcum neredeyse aynen gecer (lag yok); aykiri yutma coast'tan gelir |
| `KALMAN_CONF_GOOD` | `1.0` | 3'lu kesisim guveni (olcum gurultusu bu degere bolunur; `1.0` = tam guven) |
| `KALMAN_CONF_OK` | `0.5` | Ikili kesisim guveni (daha dusuk -> olcume daha az guven) |
| `KALMAN_WINDOW_FOLLOWS` | `True` | `True`: arama cercevesi (filtrelenmis) Kalman konumuna odaklanir; kesisimsiz karelerde coast edilmis iyi konumu takip eder -> aykiri kumelerden kurtarir |
| `USE_GPS_REVERT` | `True` | Eski "300 m geri donus" kayip-onleme. **GERCEK GPS hatasini kullanir -> GPS-denied'da gecersiz**; adil (gorsel-yalniz) kiyas icin `False` yapin. Kalman acikken zaten devre disidir |

> Sabit-konum modeli + kucuk olcum gurultusu + coast simulasyon projesinden esinlenildi;
> `MEASUREMENT_NOISE` bu hizli veri setine gore (~80 yerine ~8) ayarlandi. Baska
> platform/cozunurlukte yeniden ayarlanmasi gerekebilir.
>
> **Urgup guzergahi (216 kare, GPS'siz) sonuclari:**
>
> | Konfigurasyon | RMSE | MAE | Dogruluk(<70m) | Max hata |
> |---|---|---|---|---|
> | Gorsel-yalniz (Kalman yok, `USE_GPS_REVERT=False`) | 180 m | 46 m | %93.1 | 1280 m |
> | **Kalman ON** (varsayilan ayarlar) | **33 m** | **18 m** | **%95.4** | **202 m** |
>
> Kalman, GPS kullanmadan kaba yanlis-eslesme kumelerini yutarak hatayi ~5x dusurur.

UI ve gorunurluk odakli ayarlar (ust blok):

| Parametre | Varsayilan | Aciklama |
|-----------|-----------|----------|
| `UI_BUTTONS_ENABLED` | `True` | Ekran ustu ac/kapa butonlari |
| `UI_WINDOW_WIDTH` | `1000` | `konum` penceresi genisligi |
| `UI_WINDOW_HEIGHT` | `1000` | `konum` penceresi yuksekligi |
| `SHOW_INNER_FRAME` | `False` | Ic cerceve gorunurlugu (baslangic durumu) |
| `SHOW_ROI_FRAME` | `True` | ROI cercevesi gorunurlugu |
| `SHOW_TM_BOXES` | `True` | Template matching kutulari gorunurlugu |

---

## Kurulum

### 1. Python Ortami

Python 3.8+ onerilir. Sanal ortam olusturup bagimliliklari kurun:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install tensorflow opencv-python rasterio gdal numpy pandas pillow piexif pyproj affine
```

Alternatif olarak:

```bash
pip install -r requirements.txt
```

### 2. GDAL Kurulumu (Windows)

GDAL kurulumu Windows'ta ek adimlar gerektirebilir:

```bash
pip install GDAL
```

Sorun yasarsaniz [Christoph Gohlke'nin wheel dosyalari](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) veya `conda` kullanabilirsiniz:

```bash
conda install -c conda-forge gdal rasterio
```

### 3. CUDA GPU Destegi (Istege Bagli)

CUDA hizlandirma kullanmak icin:
- NVIDIA GPU suruculerini kurun.
- OpenCV'yi CUDA destegi ile derleyin veya `opencv-contrib-python` paketinin CUDA build'ini kullanin.
- TensorFlow GPU destegi icin uygun CUDA Toolkit ve cuDNN kurun.

### 4. Veri Hazirligi

- **Haritalar**: Georeferansli ortofoto haritalari (GeoTIFF) `haritalar/` klasorune koyun.
- **Modeller**: Egitilmis Keras model dosyalarini (`.h5`) `model/` klasorune koyun.
- **Goruntuler**: Drone goruntularini (EXIF verileri iceren JPEG) `parcalar/` klasorune koyun.
- **DEM**: Sayisal yukseklik modeli GeoTIFF dosyasini proje kok dizinine koyun.

> **Onemli**: Harita ve model dosyalari birebir eslesmelidir. Klasorlerdeki dosya sayilari esit olmalidir (ilk harita -> ilk model).

---

## Bagimliliklar

| Paket | Kullanim Amaci |
|-------|---------------|
| `tensorflow` / `keras` | Derin ogrenme model cikarimi |
| `opencv-python` (`cv2`) | Goruntu isleme, template matching, gorsellestirme |
| `rasterio` | GeoTIFF raster dosyalari okuma, CRS donusumleri |
| `gdal` (`osgeo`) | DEM (Sayisal Yukseklik Modeli) okuma |
| `numpy` | Sayisal hesaplamalar |
| `pandas` | Sonuclarin tablo formatinda yazilmasi |
| `Pillow` (`PIL`) | EXIF verisi okuma, goruntu boyutu kontrolu |
| `piexif` | EXIF meta verisi isleme |
| `pyproj` | Koordinat referans sistemi donusumleri (WGS84 <-> UTM) |
| `affine` | Afin donusum matrisi islemleri |
| `concurrent.futures` | Paralel template matching (Python standart kutuphane) |
| `multiprocessing` | Paralel islem destegi (Python standart kutuphane) |

---

## Kullanim

### Temel Calistirma

```bash
python template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py
```

### Benchmark Modu

`RUN_CFG` icinde `BENCHMARK` degerini `True` yaparak benchmark modunu etkinlestirin. Bu modda:
- Her goruntu icin arama cercevesi EXIF GPS konumuna sabitlenir.
- Adaptif takip devre disi kalir.
- Daha genis arama cercevesi (5000 px) kullanilir.

### Debug Modu

`RUN_CFG` icinde `DEBUG` degerini `True` yaparak ara goruntulerin ekranda gosterilmesini saglayabilirsiniz:
- Donduturulmus patch goruntusu
- Model cikti goruntusu

### Runtime Kontrolleri

`konum` penceresinde hem fareyle buton tiklayarak hem de klavye kisayollariyla gorunurluk ayarlari degistirilebilir:

- `T`: Trajektori ac/kapa
- `I`: Ic cerceve ac/kapa
- `O`: ROI cercevesi ac/kapa
- `R`: TM kutulari ac/kapa
- `H`: UI panelini daralt/genislet

### Loglama

Betikte `print()` tabanli mevcut ciktilar korunur. Bunlara ek olarak opsiyonel bir
`logging` altyapisi vardir (`tm` adli logger). Varsayilan seviye `WARNING` oldugundan
normal calismada **ek cikti uretmez**. Daha fazla teshis icin `RUN_CFG` icinde:

```python
"LOG_LEVEL": "INFO",   # veya "DEBUG"
"LOG_TO_FILE": True,   # tm_run.log dosyasina da yazar
```

Kod icinden kullanim: `log.info(...)`, `log.debug(...)`, `log.warning(...)`, `log.error(...)`.

### Testler

Saf (yan etkisiz) yardimci/matematik fonksiyonlari icin birim testleri `tests/`
klasorundedir. Uretim kodunu degistirmeden ana betigi modul gibi yukleyip dogrularlar.

```bash
# Proje bagimliliklarinin kurulu oldugu ortamda (orn. conda):
python -m unittest discover -s tests -v
```

Gerekli bir bagimlilik (cv2/osgeo/tensorflow) yoksa testler hata vermeden atlanir (skip).

---

## Teknik Detaylar

### EXIF Verisi Okuma

`parse_exif()` fonksiyonu PIL kutuphanesi ile EXIF verisini okur:

- **Yaw (FlightDegree)**: DJI drone'larda `MakerNote` alaninda `FlightDegree` etiketi altinda saklanir. Regex ile cikarilir ve 10'a bolunur.
- **GPS**: `GPSInfo` etiketinden enlem/boylam DMS (Derece-Dakika-Saniye) formatinda okunur ve ondalik dereceye cevrilir.
- **Irtifa**: `GPSAltitude` etiketinden metre cinsinden okunur.
- **Odak Uzakligi**: `FocalLength` etiketinden mm cinsinden okunur.
- **Kamera Modeli**: `Model` etiketinden okunarak sensor genisligi tablosunda eslestirilir.
- **Zaman Damgasi**: `DateTimeOriginal` (ve varsa `SubSecTime`) okunarak hiz hesabinda kullanilir.

### DEM Tabanli Cok Olcekli Yaklasim

Arazinin duz olmadigi durumlarda tek bir GSD degeri tum goruntuyu dogru temsil edemez. Bu sorunu cozmek icin:

1. **Merkez rakim**: Drone'un tam altindaki nokta -> `olcek_scale_test`
2. **Sol-ust kose rakimi**: 100 piksel mesafedeki sol-ust nokta -> `olcek_scale_sol_ust`
3. **Sag-alt kose rakimi**: 100 piksel mesafedeki sag-alt nokta -> `olcek_scale_sag_alt`

Her olcek icin ayri template patch uretilir:
```
olcek = (GSD / harita_cozunurlugu) x (kose_rakim / merkez_rakim)
```

Bu yaklasim, engebeli arazilerde konum tahmin dogrulugunu artirir.

### Goruntu On Isleme

1. **Yaw Duzeltmesi**: Goruntu `-yaw` kadar dondurulerek kuzey hizasina getirilir. `rotate_image()` fonksiyonu, dondurme sonrasi tum piksel verilerini koruyacak sekilde genisletilmis bir tuval uretir.

2. **Ic Dortgen Kirpma**: `largest_rotated_rect()` ve `crop_around_center()` ile dondurme sonrasi olusan siyah koseler kaldirilir.

3. **Olcekleme**: GSD oranina gore goruntu referans harita cozunurlugune getirilir. CUDA varsa `cuda_resize_if_available()` kullanilir.

4. **Patch Cikarma**: Merkezi 544x544 piksellik alanlar (+/- `fark` offset ile 3 adet) kesilir.

### Keras Model Cikarimi

Model patch'leri isleme akisi:

```
544x544 gri -> histogram esitleme -> [-1, 1] normalizasyon -> model.predict (batch=3) -> [0, 255] -> kenarlik kirpma
```

- 3 patch tek bir batch olarak islenir (verimlilik).
- Model ciktisi `squeeze` ile 2D'ye indirgenir.
- `PRED_BORDER` (16 px) kenarlik her yonden kirpilarak kenar artefaktlari onlenir.
- Sonuc: 512x512 piksellik oznitelik haritalari (gri tonlu).

### Template Matching

`cv2.TM_CCOEFF_NORMED` (Normalize Edilmis Capraz Korelasyon) yontemi kullanilir:

- **Giris**: Referans haritanin arama cercevesi bolumu + model ciktisi template
- **Cikis**: Korelasyon haritasi (her piksel icin eslestirme skoru, [-1, 1])
- **Sonuc**: `cv2.minMaxLoc` ile en yuksek korelasyon noktasi bulunur.

### Konum Belirleme (Kesisim Yontemi)

3 template'in eslestirme dortgenleri arasinda kesisim analizi yapilir:

```
Oncelik sirasi:
1. A n B n C (ucunun kesisimi)
2. A n B (ilk ikisinin kesisimi)
3. B n C (son ikisinin kesisimi)
4. A n C (ilk ve ucuncunun kesisimi)
5. Kesisim yoksa -> merkez template sonucu (B) kullanilir
```

Kesisim alaninin **merkez noktasi** nihai konum tahmini olarak kabul edilir.

### Piramit Arama (Pyramid Search)

`USE_PYRAMID = True` oldugunda iki asamali arama yapilir:

1. **Kaba Arama**: Goruntu ve template `COARSE_SCALE` (0.5) oraninda kucultulur, template matching uygulanir.
2. **Ince Arama**: Kaba aramanin buldugu konumun cevresinde (`ROI_PAD_FACTOR` genisliginde) tam cozunurlukle arama yapilir.

Bu yontem, buyuk haritalar uzerinde arama suresini **2-4 kat** azaltir.

### CUDA GPU Hizlandirma

Sistem otomatik olarak CUDA GPU varligini kontrol eder ve varsa kullanir:

| Islem | CPU | GPU (CUDA) |
|-------|-----|------------|
| Template Matching | `cv2.matchTemplate` | `cv2.cuda.createTemplateMatching` |
| Goruntu Olcekleme | `cv2.resize` | `cv2.cuda.resize` |
| Goruntu Dondurme | `cv2.warpAffine` | `cv2.cuda.warpAffine` |

- GPU'ya gecis seffaftir: hata durumunda otomatik CPU fallback.
- `match_three()`: GPU varsa goruntu bir kez GPU'ya yuklenir, 3 template sirayla eslestirilir.
- CUDA bilgisi baslangicta `log_cuda_info_once()` ile loglanir.

### Kalman Filtresi (Konum Takibi)

`USE_KALMAN=True` oldugunda, her karenin ham template-matching kesisim merkezi dogrudan
kullanilmak yerine **sabit-KONUM (constant position) Kalman filtresinden** gecirilir.
Filtre `PositionKalmanFilter` sinifidir (saf Python `math`; ek bagimlilik yok) ve
harita-piksel uzayinda calisir. Tasarim, `simulasyon` projesindeki ayni adli filtreyle
birebir aynidir.

- **Durum**: `(x, y)` (yalnizca konum). **Olcum**: `(x, y)` (kesisim merkezi).
- **NEDEN hiz durumu yok**: Once denenen sabit-HIZ modeli + innovation kapilamasi +
  pencere-takibi bu veri setinde **sapmaya** yol aciyordu — filtre yanlis hizla "coast"
  edip arama penceresini suruklUyor, olcumler bozuluyor ve hata kendini besliyordu
  (tum rotada RMSE 59 m -> 104 m). Sabit-KONUM modeli ileriye hiz EKSTRAPOLE ETMEZ;
  her gecerli olcumde olcume dogru cekilir -> **asla sapip kilitlenemez**.
- **Ongoru (predict)**: (Varsa) bilinen/komut hareketi eklenir ve belirsizlik
  `KALMAN_PROCESS_NOISE` kadar buyutulur. Offline tekrar oynatmada komut hareketi
  olmadigi icin `(0, 0)` verilir -> sadece belirsizlik buyur.
- **Confidence-olcekli olcum gurultusu (R)**: Olcum gurultusu `R / confidence` olarak
  olceklenir; guven yukseldikce olcume daha cok guvenilir. Guven kesisim seviyesinden
  gelir: 3'lu kesisim -> `KALMAN_CONF_GOOD`, ikili -> `KALMAN_CONF_OK`.
- **Kalite tabanli atlamak (coast)**: Kesisim **bulunamayan** (guvenilmez) karelerde
  `update` HIC yapilmaz; filtre son konumda kalir. Karar olcum kalitesine dayanir
  (ongoruden sapmaya degil), bu yuzden gercek bir maniv ra sirasinda iyi bir olcum
  asla yanlislikla reddedilmez.
- **Pencere-takibi**: `KALMAN_WINDOW_FOLLOWS=True` iken arama cercevesi filtrelenmis
  Kalman konumuna merkezlenir; sabit-konum modeli sapamadigindan bu guvenlidir ve
  tek-adim yanlis eslesmelere dayaniklilik saglar.
- **GPS bagimsizligi**: GPS gerektirmez (GPS-denied'a uygun). GPS hatasina dayanan eski
  "300 m geri donus" kurtarmasi yalnizca `USE_KALMAN=False` iken devrededir.
- **Hiz**: HUD/ok icin hiz, ardisik (filtrelenmis) tahmin merkezlerinin farkindan
  hesaplanir; Kalman acikken konum yumusatildigi icin hiz da daha az gurultuludur.

> Not: Kalman yalnizca `BENCHMARK=False` (adaptif takip) modunda etkindir. `USE_KALMAN=False`
> iken tum cikti ve davranis onceki surumle birebir aynidir.

---

## Koordinat Donusumleri

Sistem birden fazla koordinat sistemi arasinda donusum yapar:

| Donusum | Fonksiyon | Aciklama |
|---------|-----------|----------|
| WGS84 -> Piksel | `piksel_bul()` / `piksel_bul_fast()` | GPS koordinatini harita piksel konumuna cevirir |
| Piksel -> WGS84 | `koordinat_bul()` / `make_rc_to_ll()` | Piksel konumunu cografi koordinata cevirir |
| WGS84 -> UTM | `latlon_to_utm()` | Enlem/boylami UTM koordinatina cevirir |
| Haversine | `haversine_distance()` | Iki cografi koordinat arasi buyuk daire mesafesi |
| Quick Distance | `quick_distance()` | Yaklasik mesafe (hizli hesap) |
| Quick Distance UTM | `quick_distance_utm()` | UTM tabanli mesafe hesabi |

**CRS Donusumleri**: `pyproj.Transformer` kullanilarak EPSG:4326 (WGS84) ile haritanin/DEM'in yerel CRS'i arasinda donusum yapilir.

---

## Degerlendirme Metrikleri

Her model-harita cifti tamamlandiginda asagidaki metrikler hesaplanir:

### Mesafe Metrikleri
- **RMSE** (Root Mean Square Error): Tahmin hatalarinin karekok ortalamasi
- **MAE** (Mean Absolute Error): Mutlak hatalarin ortalamasi
- **Standart Sapma**: Hata dagiliminin yayilimi

### Siniflandirma Metrikleri (Esik: 70 metre)
- **Dogru Pozitif (TP)**: Dogru konum bulundu ve kesisim var
- **Yanlis Pozitif (FP)**: Yanlis konum bulundu ama kesisim var
- **Dogru Negatif (TN)**: Yanlis konum ve kesisim yok
- **Yanlis Negatif (FN)**: Dogru konum ama kesisim yok

### Turetilmis Metrikler
- **Hassasiyet (Precision)** = TP / (TP + FP)
- **Geri Cagirma (Recall)** = TP / (TP + FN)
- **F-Skoru** = 2 x (Precision x Recall) / (Precision + Recall)
- **Dogruluk Yuzdesi** = Dogru Tahmin / Toplam x 100

---

## Gorsellestirme

Sistem calisirken birden fazla OpenCV penceresi acilir:

### Ana Harita Penceresi ("konum")
- Referans haritanin yakinlastirilmis gorunumu
- **Kirmizi dortgen**: Template 1 (sol-ust olcek) eslestirme konumu
- **Yesil dortgen**: Template 2 (merkez olcek) eslestirme konumu
- **Mavi dortgen**: Template 3 (sag-alt olcek) eslestirme konumu
- **Siyah dortgen**: Arama cercevesi sinirlari
- **Turuncu dortgen**: ROI cercevesi (ac/kapa)
- **Sari daire**: Tahmin edilen konum
- **Yesil daire**: Gercek GPS konumu
- **Ucak ikonu**: Gercek konum ve baslik yonu
- **Sari iz / Yesil iz**: Tahmin/gercek trajektori (ac/kapa)
- **Acik mavi ok**: Hesaplanan hiz vektoru

### HUD (Head-Up Display)
- **HDG**: Yaw/baslik acisi (derece)
- **ALT**: Ucus yuksekligi (metre)
- **ERR**: Konum hatasi (metre)
- **SPD**: Hiz (`m/s` ve `km/h`)
- **Olcek Cubugu**: 100 metrelik referans cubugu
- **Arti Isareti**: Ekran merkezi

> Not: Hiz vektoru hesaplanir ve haritada ok olarak cizilir; HUD'da vektor bilesenleri gosterilmez.

### Crop vs Model Penceresi
- Ust: Kesilmis ve donduturulmus drone goruntusu (renkli)
- Alt: Keras model ciktisi (gri tonlu)

---

## Cikti Dosyalari

| Dosya | Format | Icerik |
|-------|--------|--------|
| `sonuclar.csv` | CSV | Her goruntu icin: dosya adi, sonuc (Dogru/Yanlis), gercek lat/lon, tahmin lat/lon, ucus yuksekligi |
| `sonuclar.txt` | Metin | `sonuclar.csv` ile ayni icerik, tablo formatinda |
| `modele_gore_sonuclar.txt` | Metin | Her model icin: dogru/yanlis tahmin sayisi, RMSE, MAE, std, TP, FP, TN, FN, F-skoru |

---

## Kamera Destegi

Yerlesik sensor boyutu bilinen kameralar:

| Kamera Modeli | Sensor Genisligi (mm) | Drone |
|--------------|----------------------|-------|
| L1D-20c | 13.2 | DJI Mavic 2 Pro |
| FC2204 | 6.17 | DJI Mavic 2 Zoom |

Bilinmeyen kamera modelleri icin `DEFAULT_SENSOR_WIDTH_MM` (varsayilan: 13.2 mm) kullanilir.

---

## Sinirlamalar ve Notlar

- **Harita-Model Eslesmesi**: `haritalar/` ve `model/` klasorlerindeki dosya sayilari esit olmalidir. Esit degilse minimum sayi kadar cift islenir.
- **EXIF Gereksinimi**: Goruntulerde GPS koordinatlari, irtifa ve odak uzakligi bilgileri bulunmalidir.
- **DEM Kapsami**: Islenen goruntulerin koordinatlari DEM dosyasinin kapsadigi alan icinde olmalidir.
- **Sinir Kontrolu**: Harita kenarlarina cok yakin konumlardaki goruntuler (272 piksel sinir) atlanir.
- **Referans Harita Cozunurlugu**: Varsayilan olarak ~30 cm/piksel cozunurluk kabul edilir (`29.85 cm/px`).
- **Bellek**: Buyuk ortofoto haritalar yuksek RAM tuketebilir. `OPENCV_IO_MAX_IMAGE_PIXELS` siniri `2^40`'a ayarlanmistir.
- **Yaw Destegi**: Yaw bilgisi DJI MakerNote formatinda beklenir. Diger drone ureticileri icin ozel parse gerekebilir.
- **Basari Esigi**: Varsayilan olarak 70 metre mesafe basarili konum tahmini olarak kabul edilir.
- **Adaptif Takip**: 300 metreden buyuk hata durumunda, bir onceki konuma geri donulur (kayip onleme mekanizmasi).

---

## Lisans

Bu proje, Kapadokya Universitesi tez calismasi kapsaminda gelistirilmistir.
