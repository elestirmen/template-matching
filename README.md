# UAV Görüntü Konumlandırma - Template Matching ile Derin Öğrenme Destekli Jeolokalizasyon

Bu proje, İHA (İnsansız Hava Aracı) görüntülerini **referans ortofoto harita** üzerinde otomatik olarak konumlandırmayı araştıran deneysel bir bilgisayarlı görü sistemidir. Derin öğrenme tabanlı görüntü dönüşümünü çok ölçekli template matching ile birleştirir; kontrollü GPS-merkezli benchmark ve ardışık görsel takip senaryolarını ayrı çalışma modları olarak destekler.

> **Araştırma prototipi:** Yazılım, akademik deney ve yöntem geliştirme amacı taşır. Emniyet kritik seyrüsefer veya tek başına gerçek uçuş kontrolü için doğrulanmış bir ürün değildir. Varsayılan parametreler Ürgüp çevresindeki yaklaşık 30 cm/piksel veriye özgüdür; farklı bölge, irtifa ve sensörlerde yeniden kalibrasyon gerekir.

---

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Akademik Çerçeve](#akademik-çerçeve)
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
- [Deney Protokolü ve Yeniden Üretilebilirlik](#deney-protokolü-ve-yeniden-üretilebilirlik)
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

---

## Akademik Çerçeve

### Araştırma problemi

Temel problem, bir hava görüntüsünün georeferanslı referans haritadaki konumunu görsel içerikten kestirmek ve bu kestirimi ardışık karelerde kararlı biçimde sürdürmektir. Proje özellikle şu araştırma sorularına odaklanır:

1. DEM ve kamera geometrisiyle hesaplanan GSD, sorgu ile harita arasındaki ölçek farkını ne ölçüde azaltır?
2. Öğrenilmiş görüntü dönüşümü ile klasik normalize çapraz korelasyon birlikte kullanıldığında çapraz-kaynak eşleşme yapılabilir mi?
3. Üç yükseklik/ölçek hipotezinin geometrik kesişimi tek bir eşleşmeye göre daha kararlı konum üretir mi?
4. Kalite kapılama, Kalman süzgeci ve yeniden kazanım ardışık karelerdeki yanlış sıçramaları azaltır mı?
5. Görsel konumdan ve optik akıştan elde edilen ayrı hız kanalları hangi koşullarda tutarlı sonuç verir?

### Yöntemsel katkılar

- Merkez, sol-üst ve sağ-alt DEM örneklerinden üretilen üç ölçek hipotezi
- `544 × 544 × 1` sorguların tek batch içinde Keras modeliyle dönüştürülmesi
- CPU veya uygun OpenCV-CUDA kurulumu üzerinde piramit NCC araması
- Eşleşme skorunu ve adayların geometrik yayılımını birleştiren isteğe bağlı kalite ölçümü
- Güven ağırlıklı Kalman güncellemesi, hareket kapısı ve kayıp sonrası yeniden kazanım
- Lokalizasyon hızından bağımsız, Lucas–Kanade + RANSAC tabanlı optik akış hız kestirimi

### Deney modlarının anlamı

| Mod | Arama merkezi | Uygun kullanım | Bilimsel sınır |
|---|---|---|---|
| `BENCHMARK=True` | Her karede EXIF/GPS merkezli sabit pencere | Model ve eşleştirici bileşenlerini kontrollü karşılaştırma | Uçtan uca GPS-denied otonomi sonucu değildir; gerçek konum aramayı sınırlar |
| `BENCHMARK=False` | Önceki görsel/filtrelenmiş konumu izleyen adaptif pencere | Ardışık takip ve yeniden kazanım deneyi | Başlatma, arama ve değerlendirme bilgilerinin ayrılığı ayrıca denetlenmelidir |

Adil bir GPS-denied deneyde gelecek karelerin gerçek konumu arama merkezi, hareket öncülü, kurtarma veya filtre güncellemesinde kullanılmamalıdır. Gerçek GPS yalnızca değerlendirme etiketi olarak tutulmalı ve `USE_GPS_REVERT=False` olmalıdır.

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
| `ANLIK_DIR` | `"guzergahlar/1_tezde_ucus_5"` | Drone goruntularinin bulundugu klasor |
| `DEM_PATH` | `"ana_harita_urgup_30_cm_utm_elevation.tif"` | DEM raster dosyasi yolu |
| `HARITA_DOSYALARI` | `[]` | Belirli harita dosyalari listesi (bos = klasordeki tumu) |
| `MODEL_DOSYALARI` | `[]` | Belirli model dosyalari listesi (bos = klasordeki tumu) |
| `SORT_INPUTS` | `False` | Girdi dosyalarini alfabetik sirala |
| `MAX_FRAMES` | `0` | `0`: tum kareler; `>0`: yalniz ilk N kare (hizli kontrol icin) |
| `DEFAULT_FOCAL_LENGTH_MM` | `8.8` | EXIF'te yoksa varsayilan odak uzakligi (mm) |
| `DEFAULT_SENSOR_WIDTH_MM` | `13.2` | Bilinmeyen kamera icin varsayilan sensor genisligi (mm) |
| `YAW_OFFSET_DEG` | `0.0` | EXIF yaw degerine uygulanan kalibrasyon ofseti (derece) |
| `USE_ROUTE_PROFILES` | `True` | Bilinen rotalara ait dogrulanmis profil farklarini uygula |
| `USE_GPS_ALT_REF_SIGN` | `False` | GPS altitude referans isaretini uygula |
| `WAIT_PER_MODEL` | `False` | Her model sonrasi durakla |
| `WAIT_ON_EXIT` | `False` | Program sonunda durakla |
| `LOG_LEVEL` | `"WARNING"` | Opsiyonel logger seviyesi (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Varsayilan `WARNING` oldugundan normal calismada ek cikti uretmez |
| `LOG_TO_FILE` | `False` | `True` ise loglar ayrica `tm_run.log` dosyasina yazilir |
| `MAP_RES_CM_PER_PX` | `29.85` | Referans harita cozunurlugu (cm/piksel). Olcekleme ve hiz hesabinda kullanilir |
| `KENAR_SINIR_PX` | `272` | Harita kenarina bu kadar yakin konumlar atlanir (piksel) |
| `OPTICAL_FLOW_SPEED_ENABLED` | `True` | Konum kestiriminden ayri optik-akis hiz kanalini etkinlestirir |

Kalman filtresi (konum takibi) ayarlari:

| Parametre | Varsayilan | Aciklama |
|-----------|-----------|----------|
| `USE_KALMAN` | `True` | Konum tahminini Kalman ile filtreler; `KALMAN_IN_BENCHMARK=False` iken yalniz takip modunda etkindir |
| `KALMAN_PROCESS_NOISE` | `30.0` | Surec gurultu std (px). Buyudukce olcume daha cabuk uyar (az yumusatma) |
| `KALMAN_MEASUREMENT_NOISE` | `12.0` | Olcum gurultu std (px); guven degeriyle uyarlanir |
| `KALMAN_CONF_GOOD` | `1.0` | 3'lu kesisim guveni (olcum gurultusu bu degere bolunur; `1.0` = tam guven) |
| `KALMAN_CONF_OK` | `0.5` | Ikili kesisim guveni (daha dusuk -> olcume daha az guven) |
| `KALMAN_WINDOW_FOLLOWS` | `True` | `True`: arama cercevesi (filtrelenmis) Kalman konumuna odaklanir; kesisimsiz karelerde coast edilmis iyi konumu takip eder -> aykiri kumelerden kurtarir |
| `KALMAN_REACQUIRE_FRAMES` | `1` | Bu kadar uzak ve yuksek-guvenli olcumde filtre yeniden tohumlanir |
| `KALMAN_REACQUIRE_JUMP_PX` | `700` | Olcum filtreden bu kadar uzaktaysa yeniden kazanma adayi sayilir |
| `KALMAN_STEP_GATE_MULT` | `8.0` | Adaptif hareket kapisini son karelerin medyan adimiyla olceklendirir |
| `KALMAN_MAX_STEP_PX` | `700` | Adaptif kapinin tabani; adaptif kapaliyken mutlak sinir |
| `KALMAN_USE_MOTION` | `True` | Kabul edilen olcumlerden turetilen hareketi ongoruye besler |
| `KALMAN_MOTION_EMA` | `0.2` | Hareket hizinin EMA yumusatma katsayisi |
| `KALMAN_MOTION_COAST_DECAY` | `0.7` | Coast karelerinde hareketin sonumlenme orani |
| `KALMAN_LOST_GROWTH_PX` | `800` | Kayip (surekli coast) durumunda arama penceresinin KADEMELI buyume adimi (px/kare), `CERCEVE_BOYUTU_MAX` ile sinirli. Pencere ANIDEN MAX'a ziplamaz -> tek aykiri olcum tum haritayi taratmaz. `KALMAN_COV_GATE` acikken kullanilmaz |
| `KALMAN_COV_GATE` | `False` | **COVARYANS-TABANLI ILKELI MOD.** Acikken yukaridaki ad-hoc kapilarin (`MAX_STEP`/`STEP_GATE_MULT`/`LOST_GROWTH`) YERINE kapi ve arama penceresi DOGRUDAN filtrenin kovaryansindan turetilir: innovation kapisi `= GATE_SIGMA*sqrt(P+R)`, pencere `~ 2*ROI_SIGMA*sqrt(P+R)+sablon`, process-noise (q) gercek hareket olcegine (medyan adim) baglanir. Boylece **donma/isinlama/pencere-patlamasi ucu TEK ilkeli mekanizmayla** cozulur (coast'ta P buyur -> kapi/pencere yumusakca acilir, update sonrasi toparlar). Varsayilan KAPALI; acmak icin `True` yapip RMSE'yi kiyaslayin. `KALMAN_USE_MOTION` ile birlikte en iyi |
| `KALMAN_GATE_SIGMA` | `3.0` | (COV modu) Innovation kapisi sigma: kabul esigi `= sigma*sqrt(P+R)`. Cok eliyorsa BUYUT, az eliyorsa KUCULT |
| `KALMAN_ROI_SIGMA` | `4.0` | (COV modu) Arama penceresi yari-genisligi sigma: `~ 2*sigma*sqrt(P+R)+sablon`. Gercek konumu icermesi icin `GATE_SIGMA`'dan biraz buyuk tutulur |
| `KALMAN_COV_MOTION_FRAC` | `0.4` | (COV modu) `USE_MOTION` acikken q'yu medyan adimin bu katina indirir (artik belirsizlik yalniz hizdaki SAPMA -> daha siki kapi/pencere). `USE_MOTION` kapaliyken yok sayilir |
| `KALMAN_GAIN_MAX` | `0.35` | Tek guncellemede olcume dogru uygulanabilecek azami Kalman kazanci |
| `KALMAN_LOST_SCORE` | `0.0` | `>0` ise TM skoru (`max_val2`) bunun altindaki kareler "kayip" sayilir (coast + pencere genislet). `0` = kapali. **Bu veri setinde iyi/kotu kareler ayni skor araliginda (~0.15-0.25) oldugundan ISE YARAMADI; `0`'da birakin** |
| `USE_GPS_REVERT` | `False` | Gercek GPS hatasini kullanan eski kurtarma; GPS-denied deneyde kapali kalmalidir |

> Kalman parametreleri bu veri setindeki coklu rota deneyleriyle ayarlanmistir.
> Baska platform, kare hizi veya harita cozunurlugunde yeniden kalibrasyon gerekir.
>
> **Cok-rota sonuclari (4 Urgup guzergahi, hepsi GPS'siz; RMSE, m):**
>
> | Rota | Kalman ON | Gorsel-yalniz (KF yok, revert kapali) | OFF (GPS revert koltuk degnegi) |
> |---|---|---|---|
> | 1_tezde_5 | **37** | 180 | 59 |
> | 3_tezde_7 | **231** | 329 | 202 |
> | 2_tezde_6 | **90** | 2926 | 88 |
> | 6_tezde_4 | **295** | 694 | 205 |
>
> Kalman, **GPS kullanmadan** kaba yanlis-eslesme kumelerini coast ile yutar, lock-in
> olursa yeniden-kazanim ile kurtarir. Iki rotada GPS-koltuklu OFF'a esit/ustun, zor
> rotalarda ona yakin (ama GPS gerektirmez).
>
> **Daha genis test (toplam 8 guzergah):** Kalman 5 rotada net kazandirir, 2'sinde notr
> (kolay rotada ihmal edilebilir ek maliyet), hicbirinde catastrophic degil. 2 rota
> (4_tezde_8, guz4_tezde_3) intrinsik olarak basarisiz (dogruluk ~%0-50): gorsel
> eslesmenin kendisi coker (muhtemelen harita kapsama / model uyumu) -> Kalman'in
> cozemeyecegi bir veri sorunu. `KALMAN_LOST_SCORE` denendi; bu veri icin yarar saglamadi.
>
> **Sonuc kokeni notu:** Bu sayilar kesifsel proje kayitlaridir. Tam yapilandirma,
> kaynak commit'i, veri bolunmesi, kare muhasebesi ve varlik parmak izleriyle
> eslestirilmeden yayimlanmis temel sonuc olarak kullanilmamalidir.

### Lokalizasyon kalitesi, sensor fuzyonu ve tanilama (`simulasyon` projesinden)

Asagidaki parametreler, `gps_denied_autonomy.py` modulundeki (simulasyon projesiyle
ortak, saf Python) fonksiyonlari devreye alir. **Tum bayraklar varsayilan `False`;
KAPALIYKEN mevcut davranis (Kalman dahil) BIREBIR korunur.**

Kompozit lokalizasyon kalitesi (`USE_QUALITY`) -- uc sablonun normalize skor +
**geometrik tutarliligindan** (uc kutu merkez yayilimi) [0,1] surekli bir guven uretir
ve `is_reliable` bayragiyla guvenilmez kareleri isaretler. Acikken Kalman, ikili
`1.0/0.5` guven yerine bu surekli guvenle beslenir; `is_reliable=False` kareler coast
edilir (ham skor esigi `KALMAN_LOST_SCORE`'un yerine ilkesel "kayip" tespiti).

| Parametre | Varsayilan | Aciklama |
|-----------|-----------|----------|
| `USE_QUALITY` | `False` | `True`: kompozit guven hesaplanir ve Kalman gate'ine beslenir |
| `QUALITY_SCORE_THRESHOLD` | `0.35` | Normalize skor TABANI esigi (altinda guvenilmez) |
| `QUALITY_CONFIDENCE_THRESHOLD` | `0.40` | Kompozit guven esigi (altinda guvenilmez) |
| `QUALITY_SPREAD_THRESHOLD_PX` | `120.0` | Uc kutu merkez yayilimi esigi (px); ustunde geometrik tutarsiz |

Sensor fuzyonu (`USE_FUSION`) -- ham olcumu onceki **cikti** konumuyla guvene gore
harmanlar; `FUSION_MAX_JUMP_PX*1.75`'i asan sicramalari reddeder (tek-adim yanlis
eslesmelere dayanikli). Yalnizca Kalman KAPALIYKEN ciktiyi etkiler (cift-yumusatma
olmasin); benchmark'ta da devre disidir.

| Parametre | Varsayilan | Aciklama |
|-----------|-----------|----------|
| `USE_FUSION` | `False` | `True`: ham olcum, onceki cikti konumuyla harmanlanir |
| `FUSION_BLEND_GAIN` | `0.75` | Harmanlama kazanci; efektif = `gain * confidence` |
| `FUSION_MAX_JUMP_PX` | `600.0` | Olcum priordan bu*1.75'ten uzaksa reddedilir (prior korunur) |

Tanilama (`DIAGNOSTIC_ENABLED` / `LOG_QUALITY_CSV`) -- islenen her goruntu icin
`diagnostics/diag_<ts>_m<k>/` altina **triptych PNG** (crop \| model cikisi \|
eslesen referans bolge) + `case_*_meta.json` ve dongu sonunda `summary.json` yazar;
ayrica `tani_kalite.csv`'ye kare-bazli skor/guven/yayilma/neden/hata kaydeder. Tez ve
teshis icin (ornegin yuksek-irtifa rotalarinda gorsel eslesmenin nerede coktugunu
incelemek); varsayilan KAPALI (ek I/O maliyeti). Tum I/O `try/except` ile korunur.

| Parametre | Varsayilan | Aciklama |
|-----------|-----------|----------|
| `DIAGNOSTIC_ENABLED` | `False` | `True`: kare-bazli triptych PNG + meta JSON + summary.json |
| `DIAGNOSTIC_OUTPUT_DIR` | `"diagnostics"` | Cikti kok klasoru |
| `LOG_QUALITY_CSV` | `False` | `True`: kare-bazli kalite metrikleri `tani_kalite.csv`'ye yazilir |

> Not: `USE_QUALITY` / `USE_FUSION` / `DIAGNOSTIC_ENABLED` / `LOG_QUALITY_CSV`'den
> herhangi biri acik oldugunda kalite metrigi hesaplanir; hepsi kapaliyken hic
> hesaplanmaz (sifir ek maliyet). Birim testleri: `tests/test_quality.py` (saf
> Python; agir bagimlilik gerektirmez).

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

Python 3.11 onerilir. Windows'ta GDAL/rasterio kurulumu icin Conda kullanmak daha guvenilirdir:

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

Bu yontem buyuk haritalarda arama suresini azaltabilir; hiz kazanci harita boyutu,
ROI ve donanima bagli oldugundan her deney ortaminda yeniden olculmelidir.

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
  Kalman konumuna merkezlenir; kesisimsiz karelerde coast edilmis (iyi) konumu takip
  ederek kaba aykiri kumelerinden kurtarir.
- **Yeniden-kazanim (re-acquisition)**: Pencere-takibinin nadiren yanlis bir bolgeye
  "saplanmasini" (lock-in) kirar. Yuksek-guvenli (3'lu kesisim) bir olcum filtreden
  `KALMAN_REACQUIRE_JUMP_PX`'ten uzakta ve bu durum `KALMAN_REACQUIRE_FRAMES` kez ust
  uste teyit edilirse, filtre o olcume **yeniden tohumlanir**. Ayrica kayipken (uzun
  coast / dusuk skor) arama cercevesi `CERCEVE_BOYUTU_MAX`'a genisletilir ki uzaktaki
  gercek konum yeniden gorunur olsun. Bu mekanizma, cok-rota testinde catastrophic
  bir lock-in'i (RMSE 1383 m -> 295 m) kirdi, diger rotalari bozmadan.
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

## Deney Protokolü ve Yeniden Üretilebilirlik

Akademik bir kosu icin yalnizca ortalama dogruluk veya RMSE raporlamak yeterli
degildir. Her deneyde asagidaki bilgiler birlikte saklanmalidir:

1. Git commit kimligi ve calisma agacinin temiz/kirli durumu
2. Cozumlenmis `RUN_CFG`, rota profili ve rastgelelik kaynaklari
3. Python, TensorFlow, OpenCV, GDAL, rasterio ve CUDA/cuDNN surumleri
4. Harita, model, DEM ve sorgu verilerinin kimligi; mumkunse SHA-256 degerleri
5. Denenen, kabul edilen, reddedilen, atlanan ve hata veren kare sayilari
6. Metre cinsinden medyan, MAE, RMSE, P95 ve 70 m basari orani
7. Kapsama, ardisk reddetme, yeniden kazanma suresi ve kare gecikmesi
8. Arama merkezinde veya kurtarmada kullanilan tum oracle bilgiler

`experiment_tracking.py`, deney manifesti ve kare-bazli durum muhasebesi icin
yardimci siniflar saglar. Uretilen manifest, kare CSV'si ve ozet gercekten mevcut
degilse bir kosu tamamlanmis sayilmamalidir.

> **Kosullu metrik uyarisi:** Hata yalnizca kabul edilen karelerde hesaplaniyorsa
> kapsama orani da verilmelidir. Aksi halde cok sayida zor kareyi reddeden bir
> yontem oldugundan daha basarili gorunebilir.

Veri bolme islemi mekansal blok veya bagimsiz rota duzeyinde yapilmalidir. Birbirine
cok yakin ardisk karelerin egitim ve test kumelerine dagitilmasi mekansal sizintiya
ve iyimser sonuclara yol acar.

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
- **Adaptif Takip**: Arama penceresi Kalman/kalite politikasina gore izlenir ve kayip durumunda kontrollu buyutulur. GPS tabanli eski geri donus varsayilan olarak kapalidir.
- **Benchmark Siniri**: `BENCHMARK=True`, aramayi gercek GPS cevresinde sinirladigi icin tam GPS-denied otonomi performansini olcmez.
- **Veri Setine Ozguluk**: Rota profilleri ve ayarlanmis esikler bagimsiz test verisine aktarilirken yeniden dogrulanmalidir.

---

## Atıf

Bu depo icin henuz dogrulanmis bir `CITATION.cff`, DOI veya yayimlanmis bibliyografik
kayit bulunmamaktadir. Ilgili tez/makale yayimlandiginda yazar, baslik, kurum, yil,
surum ve kalici baglanti bilgileriyle atif yapilmalidir. Bu bilgiler kesinlesmeden
tahmini bir BibTeX kaydi kullanilmamalidir.

---

## Lisans

Bu proje, Kapadokya Universitesi tez calismasi kapsaminda gelistirilmistir. Depoda
ayri bir `LICENSE` dosyasi bulunmamaktadir; kaynak kodun erisilebilir olmasi otomatik
olarak yeniden dagitim veya turev eser izni vermez. Kullanim ve lisanslama icin proje
sahipleriyle iletisime gecilmelidir.
