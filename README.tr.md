# UAV Görüntü Konumlandırma - Template Matching ile Derin Öğrenme Destekli Jeolokalizasyon

*[Click here for the English version (README.md)](README.md)*

![Python](https://img.shields.io/badge/python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-FF6F00)
![OpenCV](https://img.shields.io/badge/opencv-4.10-5C3EE8)
![Status](https://img.shields.io/badge/status-research%20prototype-yellow)
![License](https://img.shields.io/badge/license-all%20rights%20reserved-lightgrey)

Bu proje, İHA (İnsansız Hava Aracı) görüntülerini **referans ortofoto harita** üzerinde otomatik olarak konumlandırmayı araştıran deneysel bir bilgisayarlı görü sistemidir. Derin öğrenme tabanlı görüntü dönüşümünü çok ölçekli template matching ile birleştirir; kontrollü GPS-merkezli benchmark ve ardışık görsel takip senaryolarını ayrı çalışma modları olarak destekler.

> **Araştırma prototipi:** Yazılım, akademik deney ve yöntem geliştirme amacı taşır. Emniyet kritik seyrüsefer veya tek başına gerçek uçuş kontrolü için doğrulanmış bir ürün değildir. Varsayılan parametreler Ürgüp çevresindeki yaklaşık 30 cm/piksel veriye özgüdür; farklı bölge, irtifa ve sensörlerde yeniden kalibrasyon gerekir.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Akademik Çerçeve](#akademik-çerçeve)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Çalışma Akışı (Pipeline)](#çalışma-akışı-pipeline)
- [Depo Yapısı](#depo-yapısı)
- [Kod Haritası](#kod-haritası)
- [Yapılandırma (RUN\_CFG)](#yapılandırma-run_cfg)
- [Kurulum](#kurulum)
- [Bağımlılıklar](#bağımlılıklar)
- [Kullanım](#kullanım)
  - [Alternatif Arayüzler](#alternatif-arayüzler)
  - [Yeniden Üretilebilir Deney Çalıştırmaları](#yeniden-üretilebilir-deney-çalıştırmaları)
  - [Runtime Kontrolleri](#runtime-kontrolleri)
  - [Testler](#testler)
- [Teknik Detaylar](#teknik-detaylar)
  - [EXIF Verisi Okuma](#exif-verisi-okuma)
  - [DEM Tabanlı Çok Ölçekli Yaklaşım](#dem-tabanlı-çok-ölçekli-yaklaşım)
  - [Görüntü Ön İşleme](#görüntü-ön-işleme)
  - [Keras Model Çıkarımı](#keras-model-çıkarımı)
  - [Template Matching](#template-matching)
  - [Konum Belirleme (Kesişim Yöntemi)](#konum-belirleme-kesişim-yöntemi)
  - [Piramit Arama (Pyramid Search)](#piramit-arama-pyramid-search)
  - [CUDA GPU Hızlandırma](#cuda-gpu-hızlandırma)
  - [Optik Akış Hız Kestirimi](#optik-akış-hız-kestirimi)
  - [Kalman Filtresi (Konum Takibi)](#kalman-filtresi-konum-takibi)
- [Koordinat Dönüşümleri](#koordinat-dönüşümleri)
- [Değerlendirme Metrikleri](#değerlendirme-metrikleri)
- [Deney Protokolü ve Yeniden Üretilebilirlik](#deney-protokolü-ve-yeniden-üretilebilirlik)
- [Araştırma Araçları](#araştırma-araçları)
- [Görselleştirme](#görselleştirme)
- [Çıktı Dosyaları](#çıktı-dosyaları)
- [Kamera Desteği](#kamera-desteği)
- [Sınırlamalar ve Notlar](#sınırlamalar-ve-notlar)
- [Atıf](#atıf)
- [Lisans](#lisans)

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

Bu konum kanalından bağımsız olarak, bir **optik akış hız kestiricisi** ardışık kareleri izleyerek güvenilir bir eşleşme olmasa bile drone'un yer hızını raporlar. Yukarıdaki çekirdek akışın üzerine binen isteğe bağlı katmanlar (Kalman filtreleme, kompozit kalite skoru, sensör füzyonu, tanılama) her biri kendi `RUN_CFG` bayrağıyla açılıp kapanır — bkz. [Yapılandırma](#yapılandırma-run_cfg).

---

## Hızlı Başlangıç

Depoda hazır bir örnek rota (`guzergahlar/1_tezde_ucus_5`), bir harita (`haritalar/`) ve bir model (`model/`) geldiği için, bağımlılıklar kurulduktan sonra varsayılan `RUN_CFG` doğrudan çalışır:

```bash
conda create -n visual_navigation -c conda-forge python=3.11 \
    gdal=3.8.4 rasterio=1.3.10 numpy=1.26.4 pandas=2.2.2 \
    pyproj=3.6.1 affine=2.4.0 pillow=10.4.0 piexif
conda activate visual_navigation
pip install opencv-python==4.10.0.84 tensorflow==2.16.1

python template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py
```

Bu komut `konum` HUD penceresini açar, `RUN_CFG["ANLIK_DIR"]` altındaki her görüntüyü her harita/model çiftine karşı işler ve sonuçları `sonuclar.csv` / `sonuclar.txt` / `modele_gore_sonuclar.txt` dosyalarına yazar. Windows'ta GDAL kurulum notları için [Kurulum](#kurulum), kendi verinizi tanımlamak için [Yapılandırma](#yapılandırma-run_cfg), Qt ve headless çalıştırma seçenekleri için [Alternatif Arayüzler](#alternatif-arayüzler) bölümlerine bakın.

---

## Akademik Çerçeve

### Araştırma Problemi

Temel problem, bir hava görüntüsünün georeferanslı referans haritadaki konumunu görsel içerikten kestirmek ve bu kestirimi ardışık karelerde kararlı biçimde sürdürmektir. Proje özellikle şu araştırma sorularına odaklanır:

1. DEM ve kamera geometrisiyle hesaplanan GSD, sorgu ile harita arasındaki ölçek farkını ne ölçüde azaltır?
2. Öğrenilmiş görüntü dönüşümü ile klasik normalize çapraz korelasyon birlikte kullanıldığında çapraz-kaynak eşleşme yapılabilir mi?
3. Üç yükseklik/ölçek hipotezinin geometrik kesişimi tek bir eşleşmeye göre daha kararlı konum üretir mi?
4. Kalite kapılama, Kalman süzgeci ve yeniden kazanım ardışık karelerdeki yanlış sıçramaları azaltır mı?
5. Görsel konumdan ve optik akıştan elde edilen ayrı hız kanalları hangi koşullarda tutarlı sonuç verir?

### Yöntemsel Katkılar

- Merkez, sol-üst ve sağ-alt DEM örneklerinden üretilen üç ölçek hipotezi.
- `544 × 544 × 1` sorguların tek batch içinde Keras modeliyle dönüştürülmesi.
- CPU veya uygun OpenCV-CUDA kurulumu üzerinde piramit NCC araması.
- Eşleşme skorunu ve adayların geometrik yayılımını birleştiren isteğe bağlı kalite ölçümü.
- Güven ağırlıklı Kalman güncellemesi, hareket kapısı ve kayıp sonrası yeniden kazanım.
- Lokalizasyon hızından bağımsız, Lucas–Kanade + RANSAC tabanlı optik akış hız kestirimi.
- Daha sıkı bir GPS-denied değerlendirme sözleşmesini *tanımlayan*, saf Python ve birim testli bir politika katmanı (`localization_policy.py`), artı yeniden-üretilebilir çalışma manifestleri (`experiment_tracking.py`). Şu anki entegrasyon durumu için bkz. [Deney Protokolü ve Yeniden Üretilebilirlik](#deney-protokolü-ve-yeniden-üretilebilirlik).

### Deney Modlarının Anlamı

| Mod | Arama Merkezi | Uygun Kullanım | Bilimsel Sınır |
|---|---|---|---|
| `BENCHMARK=True` | Her karede EXIF/GPS merkezli sabit pencere | Model ve eşleştirici bileşenlerini kontrollü karşılaştırma | Uçtan uca GPS-denied otonomi sonucu değildir; gerçek konum aramayı sınırlar |
| `BENCHMARK=False` | Önceki görsel/filtrelenmiş konumu izleyen adaptif pencere | Ardışık takip ve yeniden kazanım deneyi | Başlatma, arama ve değerlendirme bilgilerinin ayrılığı ayrıca denetlenmelidir |

Adil bir GPS-denied deneyde gelecek karelerin gerçek konumu arama merkezi, hareket öncülü, kurtarma veya filtre güncellemesinde kullanılmamalıdır. Gerçek GPS yalnızca değerlendirme etiketi olarak tutulmalı ve `USE_GPS_REVERT=False` olmalıdır.

---

## Sistem Mimarisi

```text
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

> Kutuda tarihsel nedenlerle hâlâ `parcalar/` yazıyor; pipeline'ın gerçekte okuduğu klasör `RUN_CFG["ANLIK_DIR"]`'ın gösterdiği yerdir (varsayılan: `guzergahlar/1_tezde_ucus_5/`). `parcalar/` artık boş, eski (legacy) bir yer tutucudur — bkz. [Depo Yapısı](#depo-yapısı).

---

## Çalışma Akışı (Pipeline)

Betik çalıştırıldığında aşağıdaki adımlar sırayla gerçekleştirilir:

### 1. Başlatma ve Kaynak Yükleme
- Tüm parametreler `RUN_CFG` yapılandırma sözlüğünden (ve mevcut rotayla eşleşen `ROUTE_PROFILE_OVERRIDES` girdilerinden) okunur.
- Harita dosyaları (`haritalar/`), model dosyaları (`model/`) ve anlık görüntüler (`ANLIK_DIR`) listelenir.
- DEM (Sayısal Yükseklik Modeli) raster dosyası GDAL ile açılır.
- Referans haritanın CRS (Koordinat Referans Sistemi) bilgisi okunur ve `piksel -> koordinat` dönüştürücüsü hazırlanır.

### 2. Model ve Harita Döngüsü
Her harita-model çifti için:
- Keras modeli (`load_model`) yüklenir.
- Referans harita gri tonlu olarak OpenCV ile okunur.
- Rasterio ile haritanın CRS/transform bilgileri alınır.

### 3. Anlık Görüntü İşleme Döngüsü
Her drone görüntüsü için:

#### 3.1 EXIF Okuma
- PIL/Pillow kullanılarak EXIF verisi parse edilir.
- Yaw açısı (MakerNote -> FlightDegree), GPS koordinatları, irtifa ve odak uzaklığı çıkarılır.

#### 3.2 DEM Sorgulama ve GSD Hesabı
- Drone'un GPS konumuna karşılık gelen DEM pikselinden **arazi rakımı** okunur.
- Sol-üst ve sağ-alt köşelerdeki rakım değerleri de ayrıca alınır.
- `Uçuş Yüksekliği = GPS İrtifası - Arazi Rakımı + RAKIM_DUZELTME`
- `GSD (cm/px) = (Sensör Genişliği x Uçuş Yüksekliği x 100) / (Odak Uzaklığı x Görüntü Genişliği)`

#### 3.3 Görüntü Ön İşleme
- Görüntü **yaw açısının tersi** kadar (artı `YAW_OFFSET_DEG` kalibrasyonu) döndürülerek kuzeye hizalanır.
- Döndürülen görüntüden **en büyük iç dörtgen** kesilir (siyah köşeler kaldırılır).
- GSD oranıyla ölçeklenerek referans harita çözünürlüğüne getirilir.
- `PATCH_SIZE` (544) piksellik patch'ler üç farklı ölçekle (merkez, sol-üst, sağ-alt rakıma göre) kesilir.

#### 3.4 Derin Öğrenme Model Çıkarımı
- 3 patch, tek bir batch olarak modele verilir.
- Histogram eşitleme ve [-1, 1] normalizasyonu uygulanır.
- Model çıktısı 0-255 aralığına dönüştürülür ve kenarlık pikselleri (`PRED_BORDER`) kırpılır.

#### 3.5 Template Matching
- Referans haritanın arama çerçevesi bölümünde `cv2.TM_CCOEFF_NORMED` ile eşleştirme yapılır.
- CUDA GPU varsa `cv2.cuda.createTemplateMatching` ile hızlandırılmış eşleştirme kullanılır.
- İsteğe bağlı piramit arama: önce düşük çözünürlükte kaba arama, ardından bulunan bölgede ince arama.
- 3 template için eşzamanlı eşleştirme (`ThreadPoolExecutor`).

#### 3.6 Konum Belirleme
- 3 eşleştirme sonucunun dörtgenleri arasındaki **kesişim alanı** hesaplanır.
- Kesişim merkezinin piksel koordinatı -> coğrafi koordinat dönüşümü yapılır.
- Haversine formülü ile tahmin-gerçek arası mesafe hesaplanır.

#### 3.7 Adaptif Arama Çerçevesi
- Normal modda: bir sonraki kare için arama çerçevesi mevcut tahmine (isteğe bağlı olarak Kalman ile filtrelenmiş konuma, bkz. `KALMAN_WINDOW_FOLLOWS`) yakın bölgeye daraltılır.
- Eşleştirme başarısızsa çerçeve kademeli olarak genişletilir (`KALMAN_LOST_GROWTH_PX`, üst sınır `CERCEVE_BOYUTU_MAX`).
- Benchmark modunda: her kare için EXIF GPS merkezli sabit çerçeve kullanılır.

#### 3.8 Optik Akış Hızı (bağımsız kanal)
- Yukarıdaki sonuçtan bağımsız olarak, bu kare ile önceki kare arasında seyrek Lucas–Kanade takibi yer hızını kestirir; bkz. [Optik Akış Hız Kestirimi](#optik-akış-hız-kestirimi).

### 4. Sonuç Raporlama
- RMSE, MAE, standart sapma hesaplanır.
- Precision (hassasiyet), Recall (geri çağırma) ve F-skoru hesaplanır.
- Sonuçlar `sonuclar.csv`, `sonuclar.txt` ve `modele_gore_sonuclar.txt` dosyalarına yazılır.

---

## Depo Yapısı

Büyük ikili varlıkların (`*.tif`, `*.h5`, drone görüntüleri, sweep/run çıktıları) ve birkaç yeni yardımcı betiğin çoğu bilinçli olarak `.gitignore` ile dışlanmıştır ve yalnızca yerel çalışma kopyanızda vardır; aşağıdaki ağaç günlük çalıştığınız hâli gösterir, git'te izlenen (çok daha küçük) dosya kümesini değil.

```text
template-matching/
├── template_matching_parallel_processing_560_hizli_solust_sagalt_
│   koordinat_fonksiyonlar_icinde_cursor.py   # Ana pipeline (çalıştırılacak dosya)
├── gps_denied_autonomy.py                    # Kalite / füzyon / tanılama (import edilir)
├── optical_flow_speed.py                     # Optik akış hız kanalı (import edilir)
├── konum_ui_qt.py                            # PyQt5 kontrol paneli arayüzü
├── run_localization.py                       # Yeniden-üretilebilir çalıştırma CLI'ı (bağımsız — bkz. Kod Haritası)
├── localization_policy.py                    # GPS-denied politika kuralları (bağımsız — bkz. Kod Haritası)
├── tracking_filter.py                        # Kalman filtresi, test edilebilir kopya (bağımsız)
├── experiment_tracking.py                    # Çalışma manifesti / kare muhasebesi araçları (bağımsız)
├── requirements.txt
│
├── headless/                                 # Ekransız Linux çalıştırıcı
│   ├── README.md                             # Kurulum, systemd servisi, sorun giderme
│   ├── run_headless.py
│   └── requirements-linux.txt
├── tools/                                    # Kalman parametre-sweep + analiz betikleri
├── tests/                                    # unittest paketi (8 dosya — bkz. Testler)
│
├── haritalar/                                # Referans ortofoto harita(lar) (.tif) — HARITA_DIR
├── model/                                    # Keras model(ler) (.h5) — MODEL_DIR
├── guzergahlar/                              # Drone görüntü rotaları
│   └── 1_tezde_ucus_5/                       # Depoyla gelen örnek rota — varsayılan ANLIK_DIR
├── ana_harita_urgup_30_cm_utm_elevation.tif  # Varsayılan DEM — DEM_PATH
│                                              # (proje kökünde, yazarın kendi tekrar-çalıştırmaları için
│                                              #  ek büyük DEM/ortofoto GeoTIFF'leri de bulunur; sıfırdan bir
│                                              #  çalıştırma için yalnızca DEM_PATH ve yukarıdaki klasörler önemlidir)
│
├── results/                                  # Arşivlenmiş manuel/sweep çalıştırma çıktıları (results/kalman_sweep/, ...)
├── run_artifacts/                            # Yeniden-üretilebilir çalışma manifestleri + kare CSV'leri (aşağıya bakın)
├── diagnostics/                              # DIAGNOSTIC_ENABLED=True iken triptych PNG'ler
├── sonuclar.csv / sonuclar.txt               # Son çalıştırma: detaylı sonuçlar
├── modele_gore_sonuclar.txt                  # Son çalıştırma: model bazlı özet metrikler
│
├── arşiv/                                    # Ana betiğin arşivlenmiş bir kopyası
├── old/, top_modeller/, anlik/, anlik_t/,    # Eski / alternatif veri klasörleri (bugün çoğunlukla boş
│   parcalar/, temp/                          #  yer tutucular, geriye dönük uyumluluk için tutuluyor)
│
├── README.md / README.tr.md
└── .gitignore
```

---

## Kod Haritası

| Modül | Rolü | Durum |
|---|---|---|
| `template_matching_parallel_processing_..._cursor.py` | Ana pipeline: EXIF → DEM/GSD → ön işleme → Keras çıkarımı → template matching → kesişim → Kalman/kalite/füzyon → görselleştirme → CSV/TXT sonuçlar. | **Aktif** — çalıştırdığınız dosya budur. |
| `gps_denied_autonomy.py` | Kompozit lokalizasyon kalite skoru, sensör füzyon harmanlama, tanılama triptych/CSV yazıcıları. | Ana betiğe import edilir; `USE_QUALITY` / `USE_FUSION` / `DIAGNOSTIC_ENABLED` ile kapılanır (hepsi varsayılan `False`). |
| `optical_flow_speed.py` | Lucas–Kanade + RANSAC tabanlı, lokalizasyon kanalından bağımsız optik-akış hız kestiricisi. | Ana betiğe import edilir; `OPTICAL_FLOW_SPEED_ENABLED` ile kapılanır (varsayılan `True`). |
| `konum_ui_qt.py` | PyQt5 kontrol paneli arayüzü (arka planda `QThread` işçisi, canlı metrik kartları, aç/kapa'lar); ana betiği dinamik olarak modül gibi yükleyip çekirdek fonksiyonlarını yeniden kullanır. | Alternatif giriş noktası (`python konum_ui_qt.py`); `PyQt5` gerektirir (`requirements.txt`'te yok). |
| `headless/run_headless.py` | Aynı pipeline'ı tüm OpenCV pencereleri kapalı biçimde, ekransız Linux sunucular için çalıştırır; onun yerine periyodik olarak açıklamalı PNG'ler kaydeder. | Alternatif giriş noktası — bkz. [headless/README.md](headless/README.md). |
| `run_localization.py` | `TM_RUN_CFG_JSON` üzerinden enjekte edilen `RUN_CFG` geçersiz kılmalarıyla iki adlandırılmış, yeniden üretilebilir modu (`benchmark`, `simulation`) çalıştıran CLI. | Bağımsız, birim testli (`tests/test_run_localization.py`). Ana betik yalnızca zaten anladığı anahtarların (`BENCHMARK`, `MAX_FRAMES` vb.) alt kümesine tepki verir. |
| `localization_policy.py` | Daha sıkı bir GPS-denied protokolü için saf Python karar kuralları: hangi irtifa kaynağına güvenileceği, GPS gerçek değerinin DEM sorgusuna veya ROI değerlendirmesine ulaşıp ulaşamayacağı, adaptif ROI büyümesi, `validate_inference_policy` koruma kontrolleri. | Bağımsız, birim testli (`tests/test_localization_policy.py`). **Henüz ana pipeline'a import edilmiyor.** |
| `tracking_filter.py` | `ConstantVelocityKalmanFilter` — ana betiğe gömülü Kalman mantığının temiz, test edilebilir bir yeniden uygulaması. | Bağımsız, birim testli (`tests/test_tracking_filter.py`); ana betik kendi satır-içi Kalman uygulamasını korur. |
| `experiment_tracking.py` | Yeniden-üretilebilir çalışma manifesti oluşturucu: varlık parmak izleri, ortam/paket anlık görüntüsü, git revizyonu + kirli/temiz bayrağı, kapsama/dropout/kurtarma istatistikleriyle kare-bazlı durum muhasebesi. | Bağımsız, birim testli (`tests/test_experiment_tracking.py`); henüz ana pipeline'a import edilmiyor. |
| `tools/kalman_sweep.py`, `tools/analyze_kalman_sweep.py` | Kalman ayarlaması için parametre-sweep koşucusu (ana betiği rota/konfig başına ayrı süreç olarak çalıştırır) ve sonuç özetleyici/sıralayıcı. | Araştırma aracı, elle çalıştırılır — bkz. [Araştırma Araçları](#araştırma-araçları). |

Yukarıdaki "bağımsız" modüller ilişkili bir `simulasyon` projesinden taşınmıştır ve tek başlarına birim testlidir; bunları ana pipeline'a bağlamak devam eden bir iştir, hâlihazırda benchmark sayılarına yansımış bir şey değildir. Bunun pratikte ne anlama geldiği için bkz. [Deney Protokolü ve Yeniden Üretilebilirlik](#deney-protokolü-ve-yeniden-üretilebilirlik).

---

## Yapılandırma (RUN_CFG)

Tüm çalışma parametreleri, ana betiğin başındaki tek bir `RUN_CFG` sözlüğünden yönetilir. Bu sözlük okunduktan sonra tip-güvenli sabitlere (bool/int/float) dönüştürülür. Hemen ardından gelen ikinci, daha küçük bir sözlük olan `ROUTE_PROFILE_OVERRIDES` için bkz. [Güzergaha Özel Profil Ayarları](#güzergaha-özel-profil-ayarları).

### Temel Yollar, Arama ve Ölçek

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `BENCHMARK` | `False` | `True` ise her kare EXIF GPS merkezi etrafında sabit çerçeve kullanır (adaptif takip kapalı) |
| `DEBUG` | `False` | `True` ise ara görüntüler (patch, model çıktısı) ekranda gösterilir |
| `PATCH_SIZE` | `544` | Model giriş boyutu (piksel) |
| `PRED_BORDER` | `16` | Model çıktısından kırpılacak kenarlık (piksel) |
| `USE_PYRAMID` | `True` | Piramit (coarse-to-fine) arama etkinleştir |
| `COARSE_SCALE` | `0.5` | Piramit arama kaba ölçek faktörü |
| `ROI_PAD_FACTOR` | `0.4` | Piramit arama ince arama bölgesi genişleme katsayısı |
| `CERCEVE_BOYUTU_NORMAL` | `2048` | Normal modda arama çerçevesi boyutu (piksel) |
| `CERCEVE_BOYUTU_BENCHMARK` | `5000` | Benchmark modunda arama çerçevesi boyutu (piksel) |
| `CERCEVE_BOYUTU_MAX` | `15000` | Adaptif büyümede izin verilen azami arama çerçevesi boyutu (piksel) |
| `FARK_MAX` | `200` | Patch konumu kaydırılırken uygulanan azami piksel sınırı |
| `USE_EXIF_MOTION_SEARCH_PRIOR` | `False` | `True` ise, GPS'i çıktı olarak hiç açığa çıkarmadan, ardışık EXIF GPS sabitlerinin ima ettiği piksel hareketiyle bir sonraki arama penceresi merkezini kaydırır (`USE_GPS_REVERT`'e göre daha yumuşak bir öncül) |
| `HARITA_DIR` | `"haritalar"` | Harita dosyalarının bulunduğu klasör |
| `MODEL_DIR` | `"model"` | Keras model dosyalarının bulunduğu klasör |
| `ANLIK_DIR` | `"guzergahlar/1_tezde_ucus_5"` | Drone görüntülerinin bulunduğu klasör |
| `DEM_PATH` | `"ana_harita_urgup_30_cm_utm_elevation.tif"` | DEM raster dosyası yolu |
| `HARITA_DOSYALARI` | `[]` | Belirli harita dosyaları listesi (boş = klasördeki tümü) |
| `MODEL_DOSYALARI` | `[]` | Belirli model dosyaları listesi (boş = klasördeki tümü) |
| `SORT_INPUTS` | `False` | Girdi dosyalarını alfabetik sırala |
| `MAX_FRAMES` | `0` | `0`: tüm kareler; `>0`: yalnız ilk N kare (hızlı kontrol için) |

### EXIF, Kamera ve İrtifa

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `DEFAULT_FOCAL_LENGTH_MM` | `8.8` | EXIF'te yoksa varsayılan odak uzaklığı (mm) |
| `DEFAULT_SENSOR_WIDTH_MM` | `13.2` | Bilinmeyen kamera için varsayılan sensör genişliği (mm) |
| `CAMERA_SENSOR_BY_MODEL` | `{"L1D-20c": 13.2, "FC2204": 6.17}` | EXIF kamera modeline göre sensör genişliği geçersiz kılmaları (Mavic 2 Pro / Mavic 2 Zoom); bilinmeyen modeller `DEFAULT_SENSOR_WIDTH_MM`'e döner |
| `YAW_OFFSET_DEG` | `0.0` | EXIF yaw değerine uygulanan kalibrasyon ofseti (derece) |
| `USE_ROUTE_PROFILES` | `True` | Bilinen rotalara ait doğrulanmış profil farklarını uygula — bkz. [Güzergaha Özel Profil Ayarları](#güzergaha-özel-profil-ayarları) |
| `USE_GPS_ALT_REF_SIGN` | `False` | GPS altitude referans işaretini uygula |
| `RAKIM_DUZELTME` | `26` | Uçuş yüksekliği hesaplanırken eklenen DEM datum ofseti (metre) — GSD formülündeki `+ Düzeltme` terimi |
| `BASARI_ESIGI_KM` | `0.07` | Bu mesafenin (km) altındaki tahminler doğru sayılır (70 m) |
| `MAP_RES_CM_PER_PX` | `29.85` | Referans harita çözünürlüğü (cm/piksel). Ölçek dönüşümünde ve optik-akış piksel hızını gerçek dünya hızına çevirmede kullanılır |
| `KENAR_SINIR_PX` | `272` | Harita kenarına bu kadar yakın konumlar atlanır (piksel) |

### Trajektori ve Ekran Üzeri Arayüz

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `UI_BUTTONS_ENABLED` | `True` | Ekran üstü aç/kapa butonları ve mouse callback |
| `UI_BUTTON_FONT_SCALE` | `1.0` | Buton metin boyutu |
| `UI_BUTTON_THICKNESS` | `2` | Buton metin/çerçeve kalınlığı |
| `UI_BUTTON_SCALE` | `0.5` | Tüm UI panelinin genel ölçeği |
| `UI_WINDOW_WIDTH` | `1000` | `konum` penceresi genişliği |
| `UI_WINDOW_HEIGHT` | `1000` | `konum` penceresi yüksekliği |
| `SHOW_INNER_FRAME` | `False` | İç çerçeve görünürlüğü (başlangıç durumu) |
| `SHOW_ROI_FRAME` | `True` | ROI çerçevesi görünürlüğü (başlangıç durumu) |
| `SHOW_TM_BOXES` | `True` | Template matching kutuları görünürlüğü (başlangıç durumu) |
| `DRAW_TRAJECTORY` | `True` | `False` ise trajektori hiç çizilmez |
| `TRAJECTORY_DRAW_POINTS` | `True` | Her adımda ayrıca bir nokta işareti de çizilir |
| `TRAJECTORY_MAX_POINTS` | `0` | `0`: sınırsız; `>0`: yalnızca son N nokta tutulur |
| `TRAJECTORY_LINE_THICKNESS` | `15` | Trajektori çizgi kalınlığı |
| `TRAJECTORY_POINT_RADIUS` | `20` | Trajektori nokta işareti yarıçapı |

### Optik Akış Hız Kanalı

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `OPTICAL_FLOW_SPEED_ENABLED` | `True` | Konum kestiriminden bağımsız optik-akış hız kanalını etkinleştirir |
| `OPTICAL_FLOW_MAX_DIMENSION` | `960` | İşlem maliyetini sınırlamak için takipten önce uzun kenara uygulanan üst sınır (piksel) |
| `OPTICAL_FLOW_MAX_CORNERS` | `800` | Takip edilecek azami Lucas–Kanade köşe özelliği sayısı |
| `OPTICAL_FLOW_MIN_TRACKS` | `20` | Kestirime güvenilmesi için gereken asgari hayatta kalan iz sayısı |
| `OPTICAL_FLOW_MIN_INLIER_RATIO` | `0.45` | Kestirime güvenilmesi için gereken asgari RANSAC inlier oranı |
| `OPTICAL_FLOW_RANSAC_THRESHOLD_PX` | `2.5` | Benzerlik uydurması için RANSAC reprojeksiyon eşiği (piksel) |

### Kalman Filtresi (Konum Takibi)

`USE_KALMAN=True` olduğunda, her karenin ham template-matching kesişim merkezi doğrudan kullanılmak yerine **sabit-KONUM (constant position) Kalman filtresinden** geçirilir; harita-piksel uzayında çalışır. Tasarım, `simulasyon` projesindeki aynı adlı filtreyle birebir aynıdır.

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `USE_KALMAN` | `True` | Konum tahminini Kalman ile filtreler; yalnızca `KALMAN_IN_BENCHMARK=False` iken takip modunda etkindir |
| `KALMAN_PROCESS_NOISE` | `30.0` | Süreç gürültü std (px). Büyüdükçe ölçüme daha çabuk uyar (az yumuşatma) |
| `KALMAN_MEASUREMENT_NOISE` | `12.0` | Ölçüm gürültü std (px) — bu değerin seçilme gerekçesi için [Araştırma Araçları](#araştırma-araçları)'ndaki deneysel nota bakın |
| `KALMAN_CONF_GOOD` | `1.0` | 3'lü kesişim güveni (ölçüm gürültüsü bu değere bölünür; `1.0` = tam güven) |
| `KALMAN_CONF_OK` | `0.5` | İkili kesişim güveni (daha düşük -> ölçüme daha az güven) |
| `KALMAN_WINDOW_FOLLOWS` | `True` | `True`: arama çerçevesi (filtrelenmiş) Kalman konumuna odaklanır; kesişimsiz karelerde coast edilmiş iyi konumu takip eder -> aykırı kümelerden kurtarır |
| `KALMAN_REACQUIRE_FRAMES` | `1` | Bu kadar uzak ve yüksek-güvenli ölçümde filtre yeniden tohumlanır |
| `KALMAN_REACQUIRE_JUMP_PX` | `700` | Ölçüm filtreden bu kadar uzaktaysa yeniden kazanma adayı sayılır |
| `KALMAN_STEP_GATE_MULT` | `8.0` | Adaptif hareket kapısını son karelerin medyan adımıyla ölçeklendirir |
| `KALMAN_MAX_STEP_PX` | `700` | Adaptif kapının tabanı; adaptif kapalıyken mutlak sınır |
| `KALMAN_USE_MOTION` | `True` | Kabul edilen ölçümlerden türetilen hareketi öngörüye besler |
| `KALMAN_MOTION_EMA` | `0.2` | Hareket hızının EMA yumuşatma katsayısı |
| `KALMAN_MOTION_COAST_DECAY` | `0.7` | Coast karelerinde hareketin sönümlenme oranı |
| `KALMAN_LOST_GROWTH_PX` | `800` | Kayıp (sürekli coast) durumunda arama penceresinin kademeli büyüme adımı (px/kare), `CERCEVE_BOYUTU_MAX` ile sınırlı. `KALMAN_COV_GATE` açıkken kullanılmaz |
| `KALMAN_COV_GATE` | `False` | **Kovaryans-tabanlı ilkeli mod.** Açıkken yukarıdaki ad-hoc kapıların yerine kapı ve arama penceresi doğrudan filtrenin kovaryansından türetilir |
| `KALMAN_GATE_SIGMA` | `3.0` | (COV modu) Innovation kapısı sigma: kabul eşiği `= sigma*sqrt(P+R)`. Çok eliyorsa büyütün, az eliyorsa küçültün |
| `KALMAN_ROI_SIGMA` | `4.0` | (COV modu) Arama penceresi yarı-genişliği sigma |
| `KALMAN_COV_MOTION_FRAC` | `0.4` | (COV modu) `USE_MOTION` açıkken q'yu medyan adımın bu katına indirir |
| `KALMAN_GAIN_MAX` | `0.35` | Tek güncellemede ölçüme doğru uygulanabilecek azami Kalman kazancı |
| `KALMAN_OUTPUT_WARMUP_FRAMES` | `0` | `>0`: ilk N Kalman karesinde iç durum ısınır, ama raporlanan çıktı hâlâ ham ölçümdür |
| `KALMAN_WARMUP_RESEED` | `False` | `True`: warmup boyunca durum her karede ham ölçüme yeniden tohumlanır |
| `KALMAN_IN_BENCHMARK` | `False` | `True`: `BENCHMARK=True` iken de Kalman çıktı filtresi uygulanır |
| `KALMAN_RAW_ON_UPDATE` | `True` | `True`: kabul edilen güncellemelerde ham ölçüm, coast/aykırı karelerde Kalman çıktısı raporlanır |
| `KALMAN_LOST_SCORE` | `0.0` | `>0` ise TM skoru (`max_val2`) bunun altındaki kareler "kayıp" sayılır. `0` = kapalı. **Bu veri setinde iyi/kötü kareler aynı skor aralığında (~0.15-0.25) olduğundan işe yaramadı; `0`'da bırakın** |
| `USE_GPS_REVERT` | `False` | Gerçek GPS hatasını kullanan eski kurtarma; GPS-denied deneyde kapalı kalmalıdır |

> Kalman parametreleri bu veri setindeki çoklu rota deneyleriyle ayarlanmıştır. Başka platform, kare hızı veya harita çözünürlüğünde yeniden kalibrasyon gerekir.
>
> **Çok-rota sonuçları (4 Ürgüp güzergâhı, hepsi GPS'siz; RMSE, m):**
>
> | Rota | Kalman ON | Görsel-yalnız (KF yok, revert kapalı) | OFF (GPS revert koltuk değneği) |
> |---|---|---|---|
> | 1_tezde_5 | **37** | 180 | 59 |
> | 3_tezde_7 | **231** | 329 | 202 |
> | 2_tezde_6 | **90** | 2926 | 88 |
> | 6_tezde_4 | **295** | 694 | 205 |
>
> Kalman, **GPS kullanmadan** kaba yanlış-eşleşme kümelerini coast ile yutar, lock-in olursa yeniden-kazanım ile kurtarır. İki rotada GPS-koltuklu OFF'a eşit/üstün, zor rotalarda ona yakın (ama GPS gerektirmez).
>
> **Daha geniş test (toplam 8 güzergâh):** Kalman 5 rotada net kazandırır, 2'sinde nötr (kolay rotada ihmal edilebilir ek maliyet), hiçbirinde felaket değil. 2 rota (`4_tezde_8`, `guz4_tezde_3`) intrinsik olarak başarısız (doğruluk ~%0-50): görsel eşleşmenin kendisi çöker (muhtemelen harita kapsama / model uyumu) -> Kalman'ın çözemeyeceği bir veri sorunu — bkz. [Sınırlamalar ve Notlar](#sınırlamalar-ve-notlar)'daki yüksek-irtifa domain-shift bulgusu.
>
> **Sonuç kökeni notu:** Bu sayılar keşifsel proje kayıtlarıdır. Tam yapılandırma, kaynak commit'i, veri bölünmesi, kare muhasebesi ve varlık parmak izleriyle eşleştirilmeden yayımlanmış temel sonuç olarak kullanılmamalıdır — bkz. [Deney Protokolü ve Yeniden Üretilebilirlik](#deney-protokolü-ve-yeniden-üretilebilirlik).

### Lokalizasyon Kalitesi, Sensör Füzyonu ve Tanılama

`gps_denied_autonomy.py` modülündeki (simulasyon projesiyle ortak, saf Python) fonksiyonları devreye alan parametreler. **Tüm bayraklar varsayılan `False`; kapalıyken mevcut davranış (Kalman dahil) birebir korunur.**

Kompozit lokalizasyon kalitesi (`USE_QUALITY`) — üç şablonun normalize skor + geometrik tutarlılığından (üç kutu merkez yayılımı) [0,1] sürekli bir güven üretir ve `is_reliable` bayrağıyla güvenilmez kareleri işaretler.

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `USE_QUALITY` | `False` | `True`: kompozit güven hesaplanır ve Kalman gate'ine beslenir |
| `QUALITY_SCORE_THRESHOLD` | `0.35` | Normalize skor tabanı eşiği (altında güvenilmez) |
| `QUALITY_CONFIDENCE_THRESHOLD` | `0.40` | Kompozit güven eşiği (altında güvenilmez) |
| `QUALITY_SPREAD_THRESHOLD_PX` | `120.0` | Üç kutu merkez yayılımı eşiği (px); üstünde geometrik tutarsız |
| `NO_INTERSECTION_USE_SEARCH_CENTER` | `True` | Kesişim yoksa tekil/zayıf template kutusu yerine arama merkezi kullanılır |
| `LOW_SCORE_USE_SEARCH_CENTER` | `0.0` | `>0` ise TM skoru bu eşiğin altındayken kesişim yerine arama merkezi kullanılır |

Sensör füzyonu (`USE_FUSION`) — ham ölçümü önceki **çıktı** konumuyla güvene göre harmanlar; yalnızca Kalman kapalıyken çıktıyı etkiler (çift-yumuşatma olmasın).

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `USE_FUSION` | `False` | `True`: ham ölçüm, önceki çıktı konumuyla harmanlanır |
| `FUSION_BLEND_GAIN` | `0.75` | Harmanlama kazancı; efektif = `gain * confidence` |
| `FUSION_MAX_JUMP_PX` | `600.0` | Ölçüm priordan bu*1.75'ten uzaksa reddedilir (prior korunur) |

Tanılama (`DIAGNOSTIC_ENABLED` / `LOG_QUALITY_CSV`) — işlenen her görüntü için `diagnostics/diag_<zaman_damgası>_m<model_no>/` altına **triptych PNG** (crop \| model çıkışı \| eşleşen referans bölge) + `*_meta.json` ve döngü sonunda `summary.json` yazar; ayrıca `LOG_QUALITY_CSV=True` iken kare-bazlı skor/güven/yayılma/neden/hata bilgisini `tani_kalite_<zaman_damgası>.csv`'ye kaydeder. Tez ve teşhis için (örneğin yüksek-irtifa rotalarında görsel eşleşmenin nerede çöktüğünü incelemek); varsayılan kapalı (ek I/O maliyeti).

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `DIAGNOSTIC_ENABLED` | `False` | `True`: kare-bazlı triptych PNG + meta JSON + summary.json |
| `DIAGNOSTIC_OUTPUT_DIR` | `"diagnostics"` | Çıktı kök klasörü |
| `LOG_QUALITY_CSV` | `False` | `True`: kare-bazlı kalite metrikleri `tani_kalite_<zaman_damgası>.csv`'ye yazılır |

### Güzergaha Özel Profil Ayarları

`USE_ROUTE_PROFILES=True` iken pipeline, `ANLIK_DIR`'ın son klasör adını `RUN_CFG`'nin hemen ardından tanımlanan `ROUTE_PROFILE_OVERRIDES` sözlüğünde arar ve eşleşen `RUN_CFG` geçersiz kılmalarını uygular. Bugün üç deneysel olarak türetilmiş girdi vardır:

| Rota | Geçersiz Kılma | Neden |
|---|---|---|
| `4_tezde_ucus_8` | `YAW_OFFSET_DEG = -8.5` | Bu rotanın EXIF yaw değeri sabit bir kalibrasyon ofsetine ihtiyaç duydu (sweep ile bulundu, 2026-06-11). |
| `3_tezde_ucus_7` | `KALMAN_WINDOW_FOLLOWS = False` | Arama penceresinin Kalman tahminini takip etmesi bu rotada doğruluğu düşürdü; çıktı-yalnız filtre/hold olarak çalışması daha iyi sonuç verdi. |
| `5_tezde_ucus_9` | `KALMAN_WINDOW_FOLLOWS = False` | Yukarıdakiyle aynı gerekçe. |

Yeni bir rota otomatik olarak bir profil almaz — benzer bir kalibrasyon düzeltmesine ihtiyacı varsa kendi girdisi eklenmelidir.

### Diğer / Çalışma Zamanı

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `WAIT_PER_MODEL` | `False` | Her model sonrası duraklar |
| `WAIT_ON_EXIT` | `False` | Program sonunda duraklar |
| `LOG_LEVEL` | `"WARNING"` | Opsiyonel logger seviyesi (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Varsayılan `WARNING` olduğundan normal çalışmada ek çıktı üretmez |
| `LOG_TO_FILE` | `False` | `True` ise loglar ayrıca `tm_run.log` dosyasına yazılır |

---

## Kurulum

### 1. Python Ortamı (önerilen: Conda)

Windows'ta GDAL/rasterio'yu `pip install` ile kurmak güvenilir değildir; conda-forge çok daha sağlamdır ve projenin diğer araçları (headless çalıştırıcı) da bunu varsayar. Tek bir ortak ortam adı olan `visual_navigation`'ı kullanmak, [headless/README.md](headless/README.md) ve test paketiyle tutarlılığı korur:

```bash
conda create -n visual_navigation -c conda-forge python=3.11 \
    gdal=3.8.4 rasterio=1.3.10 numpy=1.26.4 pandas=2.2.2 \
    pyproj=3.6.1 affine=2.4.0 pillow=10.4.0 piexif
conda activate visual_navigation

pip install opencv-python==4.10.0.84   # GUI build; ekransız sunucuda opencv-python-headless kullanın
pip install tensorflow==2.16.1          # Yalnızca CPU — GPU için adım 3'e bakın
```

Platformunuz için hazır GDAL wheel'leri mevcutsa düz bir `venv` + pip de çalışır:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. GDAL Kurulumu (Windows)

GDAL kurulumu Windows'ta ek adımlar gerektirebilir. `pip install GDAL` başarısız olursa conda kullanın:

```bash
conda install -c conda-forge gdal rasterio
```

### 3. CUDA GPU Desteği (İsteğe Bağlı)

CUDA hızlandırma kullanmak için:
- NVIDIA GPU sürücülerini kurun.
- OpenCV'yi CUDA desteğiyle derleyin veya `opencv-contrib-python` paketinin CUDA build'ini kullanın.
- TensorFlow GPU desteği için uygun CUDA Toolkit ve cuDNN'i kurun.

### 4. İsteğe Bağlı Ekler

- **Qt kontrol paneli**: `pip install PyQt5` — yalnızca `konum_ui_qt.py` için gerekir, `requirements.txt`'te yoktur.
- **Headless sunucu**: `opencv-python` yerine `pip install opencv-python-headless`, ve tam sunucu kurulumu için bkz. [headless/README.md](headless/README.md) / `headless/requirements-linux.txt`.

### 5. Veri Hazırlığı

- **Haritalar**: Georeferanslı ortofoto haritaları (GeoTIFF) `haritalar/` klasörüne koyun.
- **Modeller**: Eğitilmiş Keras model dosyalarını (`.h5`) `model/` klasörüne koyun.
- **Görüntüler**: Drone görüntülerini (EXIF verisi içeren JPEG) `ANLIK_DIR`'ın gösterdiği klasöre koyun (varsayılan `guzergahlar/1_tezde_ucus_5`).
- **DEM**: Sayısal yükseklik modeli GeoTIFF dosyasını `DEM_PATH`'in gösterdiği yola koyun (varsayılan: proje kökü).

> **Önemli**: Harita ve model dosyaları birebir eşleşmelidir. Klasörlerdeki dosya sayıları eşit olmalıdır (ilk harita -> ilk model).

---

## Bağımlılıklar

| Paket | Kullanım Amacı |
|-------|---------------|
| `tensorflow` / `keras` | Derin öğrenme model çıkarımı |
| `opencv-python` (`cv2`) | Görüntü işleme, template matching, görselleştirme |
| `rasterio` | GeoTIFF raster dosyaları okuma, CRS dönüşümleri |
| `gdal` (`osgeo`) | DEM (Sayısal Yükseklik Modeli) okuma |
| `numpy` | Sayısal hesaplamalar |
| `pandas` | Sonuçların tablo formatında yazılması |
| `Pillow` (`PIL`) | EXIF verisi okuma, görüntü boyutu kontrolü |
| `piexif` | EXIF meta verisi işleme |
| `pyproj` | Koordinat referans sistemi dönüşümleri (WGS84 <-> UTM) |
| `affine` | Afin dönüşüm matrisi işlemleri |
| `concurrent.futures` | Paralel template matching (Python standart kütüphane) |
| `multiprocessing` | Paralel işlem desteği (Python standart kütüphane) |
| `PyQt5` *(isteğe bağlı)* | Yalnızca `konum_ui_qt.py` kontrol paneli arayüzü için — `requirements.txt`'te yok |

---

## Kullanım

### Temel Çalıştırma

```bash
python template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py
```

### Benchmark Modu

`RUN_CFG` içinde `BENCHMARK` değerini `True` yaparak benchmark modunu etkinleştirin. Bu modda:
- Her görüntü için arama çerçevesi EXIF GPS konumuna sabitlenir.
- Adaptif takip devre dışı kalır.
- Daha geniş arama çerçevesi (5000 px) kullanılır.

### Debug Modu

`RUN_CFG` içinde `DEBUG` değerini `True` yaparak ara görüntülerin ekranda gösterilmesini sağlayabilirsiniz (döndürülmüş patch görüntüsü, model çıktı görüntüsü).

### Alternatif Arayüzler

- **Qt kontrol paneli** — `python konum_ui_qt.py`, aynı pipeline'ı arka planda çalışan bir işçi thread'i, canlı metrik kartları ve OpenCV HUD ile aynı görünürlük aç/kapa'larına sahip bir PyQt5 penceresi arkasında çalıştırır. `PyQt5` gerektirir (`pip install PyQt5`).
- **Headless / sunucu modu** — `python headless/run_headless.py`, ekranı olmayan bir Linux sunucu için aynı pipeline'ı tüm OpenCV pencereleri kapalı biçimde çalıştırır; onun yerine periyodik olarak açıklamalı PNG kareler kaydeder. Kurulum, bir `systemd` servis örneği ve sorun giderme için bkz. [headless/README.md](headless/README.md).

### Yeniden Üretilebilir Deney Çalıştırmaları

`run_localization.py`, `RUN_CFG`'yi elle düzenlemek yerine bir çalıştırmayı iki adlandırılmış, yeniden üretilebilir moddan birine kilitleyen bir CLI sarmalayıcısıdır:

```bash
python run_localization.py --mode benchmark --max-frames 50
python run_localization.py --mode simulation --max-frames 50 --sha256-assets
```

| Mod | Ayarlar | Amaç |
|---|---|---|
| `benchmark` | `BENCHMARK=True`, `STRICT_GPS_DENIED_INFERENCE=False` | Oracle/legacy karşılaştırma — [Akademik Çerçeve](#akademik-çerçeve)'deki `BENCHMARK=True` satırı. |
| `simulation` | `BENCHMARK=False`, `STRICT_GPS_DENIED_INFERENCE=True`, `ALTITUDE_SOURCE=exif_altitude_proxy` | Daha sıkı bir GPS-denied değerlendirme için tasarlanan sözleşme — aşağıya bakın. |

Her iki mod da `WRITE_RUN_ARTIFACTS=True` ayarlar ve çözümlenmiş yapılandırmayı, ana betiğin okuyup `RUN_CFG`'ye birleştirdiği `TM_RUN_CFG_JSON` ortam değişkeni üzerinden geçirir.

> **Güncel durum:** ana pipeline yalnızca zaten anladığı anahtarların (`BENCHMARK`, `MAX_FRAMES`, …) alt kümesine tepki verir. `STRICT_GPS_DENIED_INFERENCE`, `ALTITUDE_SOURCE` ve `WRITE_RUN_ARTIFACTS` bu CLI tarafından ve `localization_policy.py`'nin politika fonksiyonları tarafından tüketilir, ama 4098 satırlık ana betik henüz bunlara göre dallanmıyor. `run_localization.py`'yi, bugün raporlanan sayıları değiştiren bir şey değil, daha sıkı protokol için tanımlanmış ve birim testli bir sözleşme olarak görün — bkz. [Deney Protokolü ve Yeniden Üretilebilirlik](#deney-protokolü-ve-yeniden-üretilebilirlik).

### Runtime Kontrolleri

`konum` penceresinde hem fareyle buton tıklayarak hem de klavye kısayollarıyla görünürlük ayarları değiştirilebilir:

- `T`: Trajektori aç/kapa
- `I`: İç çerçeve aç/kapa
- `O`: ROI çerçevesi aç/kapa
- `R`: TM kutuları aç/kapa
- `H`: UI panelini daralt/genişlet

### Loglama

Betikte `print()` tabanlı mevcut çıktılar korunur. Bunlara ek olarak opsiyonel bir `logging` altyapısı vardır (`tm` adlı logger). Varsayılan seviye `WARNING` olduğundan normal çalışmada **ek çıktı üretmez**. Daha fazla teşhis için `RUN_CFG` içinde:

```python
"LOG_LEVEL": "INFO",   # veya "DEBUG"
"LOG_TO_FILE": True,   # tm_run.log dosyasına da yazar
```

Kod içinden kullanım: `log.info(...)`, `log.debug(...)`, `log.warning(...)`, `log.error(...)`.

### Testler

```bash
python -m unittest discover -s tests -v
```

Sekiz paket, bağımlılık ağırlığına göre ayrılmış:

| Yalnızca standart kütüphane gerekir | `numpy` / `opencv` / ana modül gerekir |
|---|---|
| `test_quality.py` (`gps_denied_autonomy`'i import eder) | `test_core_functions.py` (ana betiği yükler) |
| `test_experiment_tracking.py` | `test_kalman.py` (ana betiği yükler) |
| `test_localization_policy.py` | `test_optical_flow_speed.py` (`cv2`, `numpy`) |
| `test_run_localization.py` | `test_tracking_filter.py` (`numpy`) |

`.github/workflows/unit-tests.yml` yalnızca sol sütunu, kurulum adımı olmayan çıplak bir `actions/setup-python` çalıştırıcısında koşturur — bu dördünden hiçbiri üçüncü taraf bir paket import etmez. Sağ sütun [Kurulum](#kurulum)'daki tam `visual_navigation` ortamını gerektirir. Sağ sütundaki testler `importlib.util` ile ana betiği dinamik olarak yükler; gerekli bir bağımlılık (cv2/osgeo/tensorflow) yoksa `unittest.skipIf` ile hata vermeden atlanır (skip), başarısız olmaz.

---

## Teknik Detaylar

### EXIF Verisi Okuma

`parse_exif()` fonksiyonu PIL kütüphanesi ile EXIF verisini okur:

- **Yaw (FlightDegree)**: DJI drone'larda `MakerNote` alanında `FlightDegree` etiketi altında saklanır. Regex ile çıkarılır ve 10'a bölünür.
- **GPS**: `GPSInfo` etiketinden enlem/boylam DMS (Derece-Dakika-Saniye) formatında okunur ve ondalık dereceye çevrilir.
- **İrtifa**: `GPSAltitude` etiketinden metre cinsinden okunur.
- **Odak Uzaklığı**: `FocalLength` etiketinden mm cinsinden okunur.
- **Kamera Modeli**: `Model` etiketinden okunarak sensör genişliği tablosunda eşleştirilir.
- **Zaman Damgası**: `DateTimeOriginal` (ve varsa `SubSecTime`) okunarak hız hesabında kullanılır.

### DEM Tabanlı Çok Ölçekli Yaklaşım

Arazinin düz olmadığı durumlarda tek bir GSD değeri tüm görüntüyü doğru temsil edemez. Bu sorunu çözmek için Merkez, Sol-Üst ve Sağ-Alt DEM örneklerinden türetilen üç farklı ölçek için ayrı template patch'leri üretilir. Bu yaklaşım, engebeli arazilerde konum tahmin doğruluğunu artırır.

### Görüntü Ön İşleme

1. **Yaw Düzeltmesi**: Görüntüyü kuzeye hizalar.
2. **İç Dörtgen Kırpma**: Döndürme sonrası oluşan siyah köşeleri kaldırır.
3. **Ölçekleme**: GSD'ye göre referans harita çözünürlüğüne getirir.
4. **Patch Çıkarma**: 544x544 piksellik patch'ler çıkarır.

### Keras Model Çıkarımı

Patch'leri gri tona çevirir, histogramı eşitler, [-1, 1] aralığına normalize eder ve öznitelik haritalarını çıkarmak için tek bir batch olarak Keras modelinden geçirir.

### Template Matching

Referans haritada en yüksek korelasyon noktasını bulmak için `cv2.TM_CCOEFF_NORMED` kullanır.

### Konum Belirleme (Kesişim Yöntemi)

3 template'in eşleştirme dörtgenlerinin kesişimini hesaplar. Bu kesişim alanının merkezi nihai konum tahmini olarak kabul edilir.

### Piramit Arama (Pyramid Search)

`USE_PYRAMID = True` olduğunda, büyük haritalarda süreci hızlandırmak için iki aşamalı bir arama (coarse-to-fine) kullanılır.

### CUDA GPU Hızlandırma

Sistem otomatik olarak CUDA GPU varlığını kontrol eder ve varsa Template Matching, Görüntü Ölçekleme ve Görüntü Döndürme için şeffaf biçimde kullanır.

### Optik Akış Hız Kestirimi

Konum pipeline'ından bağımsız olarak, `optical_flow_speed.py` ardışık drone kareleri arasında seyrek Lucas–Kanade özelliklerini takip eder, bunları bir RANSAC benzerlik uydurmasıyla filtreler ve inlier medyan yer değiştirmesini `MAP_RES_CM_PER_PX` ile EXIF zaman damgası farkını kullanarak gerçek dünya hızına çevirir. Varsayılan olarak etkindir (`OPTICAL_FLOW_SPEED_ENABLED`); `OPTICAL_FLOW_*` parametreleriyle (kare küçültme, köşe sayısı, asgari iz/inlier oranı, RANSAC eşiği) ayarlanabilir.

### Kalman Filtresi (Konum Takibi)

`USE_KALMAN=True` olduğunda, ham kesişim merkezi harita-piksel uzayında çalışan bir Sabit-Konum Kalman filtresiyle yumuşatılır. Bu, gürültülü ölçümleri idare eder ve GPS gerektirmeden adaptif pencere, hareket öngörüsü ve coasting'i devreye sokar. Ayrıntılı tasarım gerekçesi ve deneysel sonuçlar için [Yapılandırma (RUN_CFG)](#yapılandırma-run_cfg)'deki Kalman bölümüne bakın.

---

## Koordinat Dönüşümleri

Sistem birden fazla koordinat sistemi arasında dönüşüm yapar:

| Dönüşüm | Fonksiyon | Açıklama |
|---------|-----------|----------|
| WGS84 -> Piksel | `piksel_bul()` / `piksel_bul_fast()` | GPS koordinatını harita piksel konumuna çevirir |
| Piksel -> WGS84 | `koordinat_bul()` / `make_rc_to_ll()` | Piksel konumunu coğrafi koordinata çevirir |
| WGS84 -> UTM | `latlon_to_utm()` | Enlem/boylamı UTM koordinatına çevirir |
| Haversine | `haversine_distance()` | İki coğrafi koordinat arası büyük daire mesafesi |
| Quick Distance | `quick_distance()` | Yaklaşık mesafe (hızlı hesap) |
| Quick Distance UTM | `quick_distance_utm()` | UTM tabanlı mesafe hesabı |

**CRS Dönüşümleri**: `pyproj.Transformer` kullanılarak EPSG:4326 (WGS84) ile haritanın/DEM'in yerel CRS'i arasında dönüşüm yapılır.

---

## Değerlendirme Metrikleri

Her model-harita çifti tamamlandığında aşağıdaki metrikler hesaplanır ve `sonuclar.csv` / `sonuclar.txt` / `modele_gore_sonuclar.txt`'ye yazılır:

### Mesafe Metrikleri
- **RMSE** (Root Mean Square Error): Tahmin hatalarının karekök ortalaması
- **MAE** (Mean Absolute Error): Mutlak hataların ortalaması
- **Standart Sapma**: Hata dağılımının yayılımı

### Sınıflandırma Metrikleri (Eşik: 70 metre)
- **Doğru Pozitif (TP)**: Doğru konum bulundu ve kesişim var
- **Yanlış Pozitif (FP)**: Yanlış konum bulundu ama kesişim var
- **Doğru Negatif (TN)**: Yanlış konum ve kesişim yok
- **Yanlış Negatif (FN)**: Doğru konum ama kesişim yok

### Türetilmiş Metrikler
- **Hassasiyet (Precision)** = TP / (TP + FP)
- **Geri Çağırma (Recall)** = TP / (TP + FN)
- **F-Skoru** = 2 x (Precision x Recall) / (Precision + Recall)
- **Doğruluk Yüzdesi** = Doğru Tahmin / Toplam x 100

`run_localization.py` üzerinden çalıştırıldığında, `experiment_tracking.FrameStatusRecorder.summary()` ayrıca kapsama (`accepted / attempted`), yalnızca *kabul edilen* kareler üzerinden p50/p95/max/MAE/RMSE ve dropout istatistikleri (en uzun reddedilme serisi, kurtarma olayı sayısı ve süresi) türetir — bkz. [Deney Protokolü ve Yeniden Üretilebilirlik](#deney-protokolü-ve-yeniden-üretilebilirlik).

---

## Deney Protokolü ve Yeniden Üretilebilirlik

Akademik bir koşu için yalnızca ortalama doğruluk veya RMSE raporlamak yeterli değildir — aynı rota, yapılandırmaya, kod sürümüne ve hangi karelerin denendiğine göre çok farklı görünebilir. Bunun için iki katman vardır.

**Bugün mevcut, her zaman uygulanabilir:**
- Her raporlanan sayının yanında git commit kimliğini ve çalışma ağacının temiz/kirli durumunu saklayın.
- Çözümlenmiş `RUN_CFG`'yi (rota profili geçersiz kılmalarından sonra) ve kurulu paket sürümlerini kaydedin.
- `experiment_tracking.py` bunun yapı taşlarını sağlar: `environment_snapshot()`, `git_revision()` / `git_is_dirty()`, `file_fingerprint()` (metadata veya SHA-256 varlık özetleme) ve `build_run_manifest()`, artı her karede tam olarak bir terminal durum (`accepted` / `rejected_hold` / `skipped` / `failed`) kaydeden ve bundan kapsama, dropout-serisi ve kurtarma-süresi istatistikleri türeten bir `FrameStatusRecorder`. Görüntü/GIS bağımlılığı yoktur, bu yüzden ana çalıştırma yarıda çökse bile bir manifest yazılabilir.
- `run_artifacts/<run_id>/` (yerel, git-ignored) bu çıktının örneklerini zaten barındırıyor: `run_manifest.json` (şema, ortam, varlık parmak izleri, çözümlenmiş yapılandırma) ve model başına `frame_status_m0.csv` / `frame_summary_m0.json`.

**Tanımlanmış ve birim testli, ama henüz ana pipeline'a bağlanmamış** (bkz. [Kod Haritası](#kod-haritası)):
- `run_localization.py --mode simulation`, sıkı bir GPS-denied protokolü uygulamayı amaçlar: `t=0`'dan sonra arama merkezi, hareket önceli, DEM sorgusu veya kurtarma için gerçek GPS konumu kullanılmaz; yalnızca irtifaya, açık ve beyan edilmiş bir kaynak üzerinden güvenilir (`localization_policy.py`'deki `select_inference_altitude()`: `initial_hold`, `exif_altitude_proxy` veya `external_csv` — asla sessizce GPS sabitine düşmez).
- `localization_policy.validate_inference_policy()`, uyumsuz bir kombinasyon istendiğinde (örn. sıkı mod içinde `USE_EXIF_MOTION_SEARCH_PRIOR` veya `USE_GPS_REVERT` açılması) hata fırlatır.
- Bu kurallar `tests/test_localization_policy.py` ve `tests/test_run_localization.py` ile kapsanır, ama ana betik şu anda `localization_policy`'yi import etmiyor veya `STRICT_GPS_DENIED_INFERENCE`'a göre dallanmıyor. Bu entegrasyon tamamlanana kadar **yalnızca `BENCHMARK=False` (mevcut adaptif takip modu) gerçekten çalışan GPS-denied davranışını yansıtır** — `simulation` modunu şu an uygulanan bir garanti değil, belgelenmiş bir hedef olarak görün.

Bir tezde, makalede veya raporda bu projeden sayı alıntılarken, ikisinden hangisini kastettiğinizi açıkça belirtin.

Ek olarak, akademik bir koşuda aşağıdaki bilgiler de birlikte saklanmalıdır: Python/TensorFlow/OpenCV/GDAL/rasterio/CUDA-cuDNN sürümleri; harita, model, DEM ve sorgu verilerinin kimliği (mümkünse SHA-256); metre cinsinden medyan/MAE/RMSE/P95 ve 70 m başarı oranı; arama merkezinde veya kurtarmada kullanılan tüm oracle bilgiler. Veri bölme işlemi mekansal blok veya bağımsız rota düzeyinde yapılmalıdır — birbirine çok yakın ardışık karelerin eğitim ve test kümelerine dağıtılması mekansal sızıntıya ve iyimser sonuçlara yol açar.

> **Koşullu metrik uyarısı:** Hata yalnızca kabul edilen karelerde hesaplanıyorsa kapsama oranı da verilmelidir. Aksi halde, çok sayıda zor kareyi reddeden bir yöntem olduğundan daha başarılı görünebilir.

---

## Araştırma Araçları

`tools/kalman_sweep.py`, ana betiği rota × konfigürasyon kombinasyonu başına bir alt süreç olarak çalıştırır — `RUN_CFG`'yi `run_localization.py` ile aynı şekilde, `TM_RUN_CFG_JSON` üzerinden geçersiz kılarak — ve her çalıştırmanın sonuçlarını `results/kalman_sweep/<run_id>/` altında toplar. `tools/analyze_kalman_sweep.py` ardından bu CSV'leri birleştirir ve aday parametre kümelerini bir taban çizgisine (`off_visual_no_revert`) göre sıralar.

> **Deneysel not**, 2026-06-07 tarihli çok-rotalı bir sweep'ten: Kalman çıktısını "yalnızca-çıktı yumuşatma"ya çevirmek, o sweep'te geçerli sonuç üreten 11 rotada ortalama RMSE'yi **1073 m'den 585 m'ye** düşürdü. Bu, mevcut `KALMAN_MEASUREMENT_NOISE` varsayılanının gerekçesidir (bkz. `RUN_CFG`'de bu anahtarın üzerindeki satır içi not). Bunu kontrollü bir tez deneyi değil, yerel bir ayarlama sonucu olarak değerlendirin — alıntılamadan önce sweep'i yeniden çalıştırın.

---

## Görselleştirme

Sistem çalışırken birden fazla OpenCV penceresi açılır.

### Ana Harita Penceresi ("konum")
- Referans haritanın yakınlaştırılmış görünümü
- **Kırmızı dörtgen**: Template 1 (sol-üst ölçek) eşleştirme konumu
- **Yeşil dörtgen**: Template 2 (merkez ölçek) eşleştirme konumu
- **Mavi dörtgen**: Template 3 (sağ-alt ölçek) eşleştirme konumu
- **Siyah dörtgen**: Arama çerçevesi sınırları
- **Turuncu dörtgen**: ROI çerçevesi (aç/kapa)
- **Sarı daire**: Tahmin edilen konum
- **Yeşil daire**: Gerçek GPS konumu
- **Uçak ikonu**: Gerçek konum ve başlık yönü
- **Sarı iz / Yeşil iz**: Tahmin/gerçek trajektori (aç/kapa)
- **Açık mavi ok**: Hesaplanan hız vektörü

### HUD (Head-Up Display)
- **HDG**: Yaw/başlık açısı (derece)
- **ALT**: Uçuş yüksekliği (metre)
- **ERR**: Konum hatası (metre)
- **SPD**: Hız (`m/s` ve `km/h`)
- **Ölçek Çubuğu**: 100 metrelik referans çubuğu
- **Artı İşareti**: Ekran merkezi

Qt arayüzü (`konum_ui_qt.py`) aynı bilgiyi ek canlı metrik kartlarıyla bir PyQt5 penceresi içinde gösterir.

> Not: Hız vektörü hesaplanır ve haritada ok olarak çizilir; HUD'da vektör bileşenleri gösterilmez.

### Crop vs Model Penceresi
- Üst: Kesilmiş ve döndürülmüş drone görüntüsü (renkli)
- Alt: Keras model çıktısı (gri tonlu)

---

## Çıktı Dosyaları

| Dosya | Format | İçerik |
|-------|--------|--------|
| `sonuclar.csv` | CSV | Her görüntü için: dosya adı, sonuç (Doğru/Yanlış), gerçek lat/lon, tahmin lat/lon, uçuş yüksekliği |
| `sonuclar.txt` | Metin | `sonuclar.csv` ile aynı içerik, tablo formatında |
| `modele_gore_sonuclar.txt` | Metin | Her model için: doğru/yanlış tahmin sayısı, RMSE, MAE, std, TP, FP, TN, FN, F-skoru |
| `run_artifacts/<run_id>/` *(git-ignored)* | JSON/CSV | `run_manifest.json` + model başına `frame_status_m*.csv` / `frame_summary_m*.json`; yalnızca `run_localization.py` üzerinden `WRITE_RUN_ARTIFACTS` ayarlandığında |
| `diagnostics/diag_<zaman_damgası>_m<model_no>/` *(git-ignored)* | PNG/JSON | `DIAGNOSTIC_ENABLED=True` iken kare-bazlı triptych PNG'ler + meta JSON, artı koşu sonunda `summary.json` |
| `tani_kalite_<zaman_damgası>.csv` | CSV | `LOG_QUALITY_CSV=True` iken kare-bazlı kalite metrikleri |
| `results/` *(git-ignored)* | — | Arşivlenmiş manuel/sweep çalıştırmaları, `results/kalman_sweep/` dahil |
| `tm_run.log` | Metin | `LOG_TO_FILE=True` iken opsiyonel logger çıktısı |

---

## Kamera Desteği

`CAMERA_SENSOR_BY_MODEL`'de EXIF kamera modeline göre sensör genişlikleri tanımlıdır:

| Kamera Modeli | Sensör Genişliği (mm) | Drone |
|--------------|----------------------|-------|
| `L1D-20c` | 13.2 | DJI Mavic 2 Pro |
| `FC2204` | 6.17 | DJI Mavic 2 Zoom |

Bilinmeyen kamera modelleri için `DEFAULT_SENSOR_WIDTH_MM` (varsayılan: 13.2 mm) kullanılır; EXIF'te odak uzaklığı hiç yoksa `DEFAULT_FOCAL_LENGTH_MM` (8.8 mm) kullanılır.

---

## Sınırlamalar ve Notlar

- **Harita-Model Eşleşmesi**: `haritalar/` ve `model/` klasörlerindeki dosya sayıları eşit olmalıdır (her iterasyonda bir Keras modeli bir referans haritayla eşleştirilir).
- **EXIF Gereksinimi**: Görüntülerde GPS koordinatları, irtifa ve odak uzaklığı bilgileri bulunmalıdır (yaw için MakerNote `FlightDegree`).
- **DEM Kapsamı**: İşlenen görüntülerin koordinatları DEM dosyasının kapsadığı alan içinde olmalıdır.
- **Bellek**: Büyük ortofoto haritalar ve DEM'ler (çoklu GB GeoTIFF) yüksek RAM tüketebilir.
- **Başarı Eşiği**: 70 metrelik bir eşik (`BASARI_ESIGI_KM=0.07`) toplu metriklerde bir tahmini "doğru" olarak sınıflandırmak için kullanılır.
- **Bilinen yüksek-irtifa arıza modu:** Yaklaşık ≥1000 m AGL'de uçulan rotalar (örn. `4_tezde_ucus_8`) şu anda zayıf performans gösteriyor. Bu, bir arama-kapsamı veya ölçek-hesabı hatasından ziyade bir **model domain-shift sorunu** olarak karakterize edildi — Keras öznitelik-çıkarım modeli bu irtifa bandındaki görüntülerle eğitilmemişti; `YAW_OFFSET_DEG`'i yeniden kalibre etmek (bkz. [Güzergaha Özel Profil Ayarları](#güzergaha-özel-profil-ayarları)) bunu düzeltmiyor. Muhtemel çözüm, bir `RUN_CFG` değişikliği değil, daha yüksek irtifadaki görüntülerle yeniden eğitim/fine-tuning'dir.
- `ROUTE_PROFILE_OVERRIDES` şu anda üç belirli rota için deneysel kalibrasyon düzeltmelerini sabit kodlar; yeni bir rota kendi girdisine ihtiyaç duyabilir.
- Varsayılan parametreler Ürgüp çevresindeki ~30 cm/px görüntüler için ayarlanmıştır; farklı bir sensör, bölge veya irtifa bandı için `MAP_RES_CM_PER_PX`, `DEFAULT_SENSOR_WIDTH_MM` / `CAMERA_SENSOR_BY_MODEL` ve `RAKIM_DUZELTME`'yi yeniden kalibre etmeyi bekleyin.
- Hâlihazırda çalışan GPS-denied davranışı ile daha sıkı, tanımlanmış-ama-henüz-bağlanmamış `simulation` protokolü arasındaki fark için bkz. [Deney Protokolü ve Yeniden Üretilebilirlik](#deney-protokolü-ve-yeniden-üretilebilirlik).

---

## Atıf

Bu depo için henüz doğrulanmış bir `CITATION.cff`, DOI veya yayımlanmış bibliyografik kayıt bulunmamaktadır. İlgili tez/makale yayımlandığında yazar, başlık, kurum, yıl, sürüm ve kalıcı bağlantı bilgileriyle atıf yapılmalıdır. Bu bilgiler kesinleşmeden tahmini bir BibTeX kaydı kullanılmamalıdır.

---

## Lisans

Bu proje MIT Lisansı ve Apache Lisansı 2.0 altında çift lisanslıdır.
