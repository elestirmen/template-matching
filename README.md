# UAV Görüntü Konumlandırma

Derin öğrenme destekli template matching ile drone/İHA görüntülerini referans harita üzerinde konumlandıran çalışma.

Bu repo, tek bir ana betik etrafında çalışır:

- `template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py`

## Ne Yapar?

Sistem, her anlık görüntü için şu adımları yapar:

1. EXIF'ten yaw, GPS, irtifa, odak uzaklığı ve (varsa) zaman bilgisini okur.
2. DEM üzerinden arazi yüksekliğini hesaplar.
3. GSD/ölçek düzeltmesi ile görüntüyü harita çözünürlüğüne yaklaştırır.
4. 3 farklı ölçekli template üretir (sol-üst, merkez, sağ-alt).
5. Keras modelinden özellik haritası çıkarır.
6. Haritada template matching yapar (CPU/CUDA).
7. 3 eşleşmenin kesişiminden konum tahmini üretir.
8. Tahmini konum ile GPS konumu arasındaki hatayı raporlar.

Ayrıca:

- Trajektori çizimi
- ROI çerçevesi görünürlüğü
- Hız hesaplama + hız vektörü oku çizimi
- HUD'da hız (`m/s` ve `km/h`) gösterimi

Not: Hız vektörü hesaplanır ve haritada ok olarak çizilir; HUD'da vektör bileşenleri gösterilmez.

---

## Hızlı Başlangıç

### 1) Python ortamı

Python 3.9+ önerilir.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
```

### 2) Paketleri kur

```bash
pip install tensorflow opencv-python rasterio gdal numpy pandas pillow piexif pyproj affine
```

Windows'ta `gdal` kurulumu zorlanırsa `conda-forge` tercih edin:

```bash
conda install -c conda-forge gdal rasterio
```

### 3) Girdileri yerleştir

- Haritalar: `haritalar/`
- Modeller: `model/`
- Anlık görüntüler: `parcalar/`
- DEM dosyası: proje kökünde (varsayılan: `ana_harita_urgup_30_cm_utm_elevation.tif`)

### 4) Çalıştır

```bash
python template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py
```

---

## Minimum Klasör Yapısı

```text
template matching/
├─ template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py
├─ README.md
├─ haritalar/
├─ model/
├─ parcalar/
└─ ana_harita_urgup_30_cm_utm_elevation.tif
```

---

## Runtime UI Kontrolleri

Ana pencere: `konum`

Klavye kısayolları:

- `T`: Trajektori aç/kapat
- `I`: İç çerçeve aç/kapat
- `O`: ROI çerçevesi aç/kapat
- `R`: TM kutuları aç/kapat
- `H`: UI panel daralt/genişlet

Mouse ile sol üstteki butonlara tıklayarak da aynı kontroller yapılabilir.

---

## Ekranda Ne Görürsünüz?

### Harita üzerine çizilenler

- Tahmini konum (sarı nokta)
- Gerçek konum (yeşil nokta)
- Uçak ikonu (heading/yaw yönü)
- Eşleşme kutuları (RGB)
- Arama iç çerçevesi
- ROI çerçevesi (turuncu)
- Hız vektörü oku (açık mavi)

### HUD

- `HDG`: heading/yaw
- `ALT`: uçuş yüksekliği
- `ERR`: konum hatası (metre)
- `SPD`: hız (`m/s` ve `km/h`)

Ek olarak:

- 100 m ölçek çubuğu
- Merkez crosshair

---

## Konfigürasyon (RUN_CFG)

Betikte **iki adet `RUN_CFG` bloğu** bulunur.

- Üst blok: UI ve bazı genel ayarların ilk değerleri
- Orta bölümdeki ikinci blok: çekirdek eşleştirme parametrelerini tekrar set eder

Pratikte, bakım yaparken iki bloğun da tutarlı tutulması gerekir.

### Kritik parametreler

| Parametre | Açıklama |
|---|---|
| `BENCHMARK` | Her karede sabit GPS merkezli arama (adaptif takip yerine) |
| `DEBUG` | Ara görselleri gösterir |
| `PATCH_SIZE` | Model giriş patch boyutu |
| `PRED_BORDER` | Model çıktısından kırpılan kenar |
| `USE_PYRAMID` | Coarse-to-fine template matching |
| `COARSE_SCALE` | Piramit kaba arama ölçeği |
| `ROI_PAD_FACTOR` | İnce arama ROI genişliği |
| `CERCEVE_BOYUTU_NORMAL` | Normal mod arama çerçevesi |
| `CERCEVE_BOYUTU_BENCHMARK` | Benchmark mod arama çerçevesi |
| `HARITA_DIR` / `MODEL_DIR` / `ANLIK_DIR` | Girdi klasörleri |
| `DEM_PATH` | DEM raster yolu |
| `HARITA_DOSYALARI` / `MODEL_DOSYALARI` | Dosya seçimini elle sınırla |
| `SORT_INPUTS` | Girdileri sırala |
| `DEFAULT_FOCAL_LENGTH_MM` | EXIF yoksa odak uzaklığı fallback |
| `DEFAULT_SENSOR_WIDTH_MM` | Kamera modeli bilinmiyorsa sensör fallback |
| `USE_GPS_ALT_REF_SIGN` | GPS altitude işaret düzeltmesi |
| `WAIT_PER_MODEL`, `WAIT_ON_EXIT` | Çalışma sonunda bekleme seçenekleri |

### UI parametreleri (üst RUN_CFG bloğu)

| Parametre | Açıklama |
|---|---|
| `UI_BUTTONS_ENABLED` | UI toggle paneli aktif/pasif |
| `UI_BUTTON_FONT_SCALE`, `UI_BUTTON_THICKNESS`, `UI_BUTTON_SCALE` | Buton görünümü |
| `UI_WINDOW_WIDTH`, `UI_WINDOW_HEIGHT` | Pencere başlangıç boyutu |
| `SHOW_INNER_FRAME` | İç çerçeve başlangıç durumu |
| `SHOW_ROI_FRAME` | ROI çerçeve başlangıç durumu |
| `SHOW_TM_BOXES` | TM kutuları başlangıç durumu |
| `DRAW_TRAJECTORY` | Trajektori başlangıç durumu |

### Uzantı filtreleri (opsiyonel)

Kodda bu alanlar desteklenir; `RUN_CFG` içine ekleyebilirsiniz:

- `HARITA_UZANTILARI` (varsayılan: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.jp2`)
- `MODEL_UZANTILARI` (varsayılan: `.h5`, `.keras`, `.hdf5`)
- `ANLIK_UZANTILARI` (varsayılan: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`)

---

## Hız Hesabı Nasıl Yapılıyor?

- Her karede tahmini konum pikseli alınır.
- Ardışık iki tahmin arasındaki farktan hız vektörü hesaplanır.
- Piksel -> metre dönüşümü için `mekansal_cozunurluk` (`cm/px`) kullanılır.
- Zaman farkı:
  - EXIF timestamp varsa onun farkı,
  - yoksa wall-clock farkı (çalışma zamanı) kullanılır.
- Sonuç:
  - `speed_mps`: hız büyüklüğü
  - `speed_vx_mps`, `speed_vy_mps`: vektör bileşenleri (HUD'da gizli)
  - Harita üstünde ok olarak çizim

---

## Çalışma Modları

### Normal mod (`BENCHMARK=False`)

- İlk karede EXIF GPS merkezli arama yapılır.
- Sonraki karelerde önceki tahmine yakın bölgede arama daraltılır (adaptif takip).
- Takip kayarsa çerçeve boyutu artırılarak toparlama denenir.

### Benchmark mod (`BENCHMARK=True`)

- Her kare EXIF GPS merkezli sabit arama çerçevesi kullanır.
- Adaptif takip devre dışıdır.
- Karşılaştırmalı performans testleri için uygundur.

---

## Teknik Pipeline (Detay)

1. `parse_exif()` ile yaw/GPS/irtifa/odak/zaman okunur.
2. DEM üzerinden merkez + köşe rakımları alınır.
3. Görüntü `-yaw` ile döndürülüp siyah köşeler kırpılır.
4. GSD oranı ile ölçeklenir.
5. 3 farklı ölçekli patch hazırlanır.
6. Patch'ler modelde batch olarak işlenir.
7. Her patch için template matching yapılır.
8. Kesişim yöntemi ile tek konum tahmini çıkarılır.
9. Piksel -> coğrafi koordinat dönüşümü ve hata hesaplanır.
10. Görselleştirme + HUD + dosya çıktıları üretilir.

---

## Değerlendirme Metrikleri

Kod tarafında şu metrikler üretilir:

- RMSE
- MAE
- Standart sapma
- TP / FP / TN / FN
- Precision / Recall / F-score
- Doğru-yanlış tahmin adetleri

Başarı eşiği: yaklaşık **70 m** (`0.07 km`).

---

## Kamera Notu

Kamera modeline göre sensör genişliği fallback tablosu:

| Model | Sensör Genişliği |
|---|---|
| `L1D-20c` | `13.2 mm` |
| `FC2204` | `6.17 mm` |

Model bilinmiyorsa `DEFAULT_SENSOR_WIDTH_MM` kullanılır.

---

## Çıktılar

| Dosya | İçerik |
|---|---|
| `sonuclar.csv` | Kare bazında doğruluk, gerçek/tahmin koordinatları, uçuş yüksekliği |
| `sonuclar.txt` | Aynı bilgilerin metin tablosu |
| `modele_gore_sonuclar.txt` | Model bazlı özet metrikler (RMSE, MAE, std, TP/FP/TN/FN, F-score) |

---

## Sorun Giderme

### 1) `Harita klasoru bulunamadi` / `Model klasoru bulunamadi`

`RUN_CFG` içindeki klasör yollarını kontrol edin.

### 2) `EXIF okunamadi` veya GPS/irtifa eksik

Görüntülerde EXIF metadata bulunduğundan emin olun.

### 3) `DEM disinda kalan koordinat`

DEM kapsama alanı, görüntülerin GPS bölgesini içermelidir.

### 4) Hata çok yüksek / takip kayıyor

- `CERCEVE_BOYUTU_NORMAL` değerini artırın.
- `USE_PYRAMID` açık kalsın.
- `ROI_PAD_FACTOR` ile ince arama alanını artırın.
- Gerekirse benchmark modu ile karşılaştırmalı test yapın.

### 5) CUDA görünmüyor

- OpenCV CUDA derlemesi yüklü olmayabilir.
- NVIDIA sürücü + CUDA/cuDNN uyumunu kontrol edin.
- Kod CPU fallback ile çalışmaya devam eder.

---

## Performans Notları

- Büyük ortofotolar yüksek RAM tüketebilir.
- Piramit arama (`USE_PYRAMID`) genelde ciddi hız kazandırır.
- GPU varsa template matching ve resize hızlanır.

---

## Sınırlamalar

- Yaw parse mantığı DJI MakerNote formatına odaklıdır.
- Harita/model eşleşmesinde sıralama önemli olabilir.
- Çok kenar bölgelerde (harita sınırına yakın) kareler atlanabilir.

---

## Lisans / Akademik Not

Bu çalışma Kapadokya Üniversitesi tez kapsamındaki geliştirmelerden türetilmiştir.
