# Headless (ekransiz) Linux çalıştırma

Bu klasör, projeyi **ekrana hiçbir pencere açmadan** (GUI'siz) bir Linux sunucuda
çalıştırmak içindir. **Ana kod değiştirilmez** — buradaki `run_headless.py` ana
işleme modülünü üst dizinden import eder ve ona boş olmayan bir `callbacks`
sözlüğü geçer. Ana hattaki tüm pencere kodu zaten `if callbacks is None:` ile
korunduğundan tamamen atlanır.

Çıktılar:
- **Konsol/log:** per-frame özet satırları (stdout, istenirse `tm_run.log`).
- **CSV/TXT:** proje kökündeki `sonuclar.csv`, `sonuclar.txt`,
  `modele_gore_sonuclar.txt`.
- **PNG kareler (opsiyonel):** her N karede bir, bilgi paneli bindirilmiş
  görüntü `headless/headless_out/` içine.

---

## 1) Kurulum (conda — önerilen)

GDAL/rasterio'yu pip ile derlemek Linux'ta sorunludur; conda-forge en sağlamı:

```bash
# Mamba/conda
conda create -n visual_navigation -c conda-forge python=3.11 \
    gdal=3.8.4 rasterio=1.3.10 numpy=1.26.4 pandas=2.2.2 \
    pyproj=3.6.1 affine=2.4.0 pillow=10.4.0 piexif
conda activate visual_navigation

# OpenCV (GUI'siz) ve TensorFlow pip ile:
pip install opencv-python-headless==4.10.0.84
pip install tensorflow==2.16.1            # CPU-only
# GPU'lu sunucu icin yerine:
# pip install "tensorflow[and-cuda]==2.16.1"   # CUDA 12.3 / cuDNN 8.9 gerekir
```

> Eğer `opencv-python` (GUI'li) zaten kuruluysa kaldırın:
> `pip uninstall -y opencv-python` — `opencv-python-headless` ile birlikte
> bulunmamalı.

## 1-alt) Kurulum (yalnızca pip — conda yoksa)

```bash
sudo apt-get update && sudo apt-get install -y gdal-bin libgdal-dev
pip install -r headless/requirements-linux.txt
```

---

## 2) Veriyi sunucuya taşı

`RUN_CFG` göreli yolları **proje köküne** göre çözer. Şu klasör/dosyalar kökte
bulunmalı (büyük/küçük harf Linux'ta **duyarlı**, birebir aynı olmalı):

- `haritalar/` (referans haritalar)
- `model/` (Keras modelleri: `.h5/.keras`)
- `guzergahlar/...` (işlenecek anlık görüntüler — `ANLIK_DIR`)
- DEM `.tif` (`RUN_CFG["DEM_PATH"]`, ör. `ana_harita_urgup_30_cm_utm_elevation.tif`)

`.tif`/DEM dosyaları GB boyutunda; `rsync` ile aktarın:

```bash
rsync -avP haritalar/ model/ guzergahlar/ ana_harita_urgup_30_cm_utm_elevation.tif \
    kullanici@sunucu:/opt/visual_navigation/template-matching/
```

---

## 3) Çalıştırma

Proje kökünden veya bu klasörden çalıştırılabilir (yollar köke göre çözülür):

```bash
# Varsayilan: her 10 karede bir PNG + konsol ozetleri
python headless/run_headless.py

# Sadece konsol + CSV (hic kare kaydetme) — en hizli
python headless/run_headless.py --no-frames --log-file tm_run.log --log-level INFO

# Belirli bir guzergah ve daha sik kare kaydi
python headless/run_headless.py --anlik guzergahlar/guzergah_2_tezde_ucus_1 --frame-every 5
```

Argümanlar:
| Argüman | Açıklama | Varsayılan |
|---|---|---|
| `--out-dir` | PNG kayıt klasörü | `headless/headless_out` |
| `--frame-every N` | Her N karede bir PNG | `10` |
| `--no-frames` | Hiç kare kaydetme | kapalı |
| `--frame-max-width` | Kayıtta maks. genişlik (px), 0=kapalı | `1600` |
| `--log-file` | Ek logger dosyası (LOG_TO_FILE) | yok |
| `--log-level` | DEBUG/INFO/WARNING/ERROR | RUN_CFG |
| `--anlik` / `--model` / `--harita` | İlgili `*_DIR`'i ez | RUN_CFG |

> Diğer ayarlar (DEM, Kalman, eşikler vb.) ana dosyadaki `RUN_CFG` üzerinden;
> burada ezmek istemiyorsanız dokunmanıza gerek yok.

---

## 4) Arka planda / servis olarak

```bash
# Basit: nohup
nohup python headless/run_headless.py --no-frames > headless/run.out 2>&1 &
tail -f headless/run.out
```

systemd örneği (`/etc/systemd/system/visnav.service`):

```ini
[Unit]
Description=Visual Navigation (headless)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/visual_navigation/template-matching
ExecStart=/opt/conda/envs/visual_navigation/bin/python headless/run_headless.py --no-frames --log-file tm_run.log
Restart=on-failure
# GPU'lu sunucuda gerekirse:
# Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now visnav
journalctl -u visnav -f
```

---

## 5) Doğrulama (smoke test)

1. Küçük bir `guzergah` ile çalıştır: birkaç karede `[i/N] ERR=... ACC=...`
   satırları akmalı, **hiçbir pencere açılmamalı**, hata vermemeli.
2. Bitince `sonuclar.csv` ve `modele_gore_sonuclar.txt` güncellenmiş olmalı.
3. `--frame-every` verdiyseniz `headless/headless_out/frame_*.png` oluşmalı.

## Sorun giderme

- **`cv2.error: ... not implemented` / `imshow`** → GUI'li OpenCV kurulu veya bir
  GUI çağrısı tetiklenmiş. `opencv-python` kaldırılıp `opencv-python-headless`
  kurulmalı. Bu runner GUI çağrısı yapmaz; `DEBUG=False` olduğundan emin olun.
- **`FileNotFoundError: Ana isleme dosyasi bulunamadi`** → runner'ı taşıdıysanız
  `TM_PROJECT_ROOT=/proje/koku python run_headless.py` ile kökü belirtin.
- **GDAL/rasterio import hatası** → conda-forge ile kurun (Bölüm 1).
- **Klasör bulunamadı (`Harita/Model/Anlik klasoru bulunamadi`)** → Linux harf
  duyarlılığı; `haritalar/model/guzergahlar` adlarını birebir kontrol edin.
