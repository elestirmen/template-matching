# Mission Control — Template Matching

Qt arayüzü Visual ve TERCOM uygulamalarıyla aynı `mission_ui` kaynak paketini
kullanır. Üç repodaki şablon dosyaları birebir aynıdır; her repo kendi kopyasıyla
bağımsız çalışır. Ortak paket uygulamanın seçtiği Qt bağlayıcısını kullanır:
bu uygulama PyQt5, diğer iki uygulama PySide6 kullanır.

## Başlatma

Windows'ta `baslat_ui.cmd` dosyasını açın. Başlatıcı mevcut
`%USERPROFILE%\anaconda3` kurulumundaki `visual_navigation` ortamını etkinleştirir.
Alternatif olarak Anaconda Prompt'ta bu klasöre geçip çalıştırın:

```powershell
conda activate visual_navigation
python konum_ui_qt.py
```

PyQt5 yalnızca masaüstü arayüzü için gereklidir. Yeni kurulumlarda
`python -m pip install -r requirements-ui.txt` kullanılabilir; mevcut çalışan
Conda/CUDA ortamını yeniden kurmaya gerek yoktur.

## Düzen

- Üst: başlat/durdur, Kalman ve ayrı görünüm kontrolleri.
- Sol: gözlem ve sinir ağı çıktısı; sürüklenebilir panel sınırları.
- Orta: kaynak görüntüden yeniden ölçeklenen operasyon haritası.
- Sağ: temel metrikler, kare ilerlemesi, telemetri ve olay günlüğü sekmeleri.
- LOC SPD eşleştirme/Kalman konumlarından, OF SPD optik akıştan gelir.
  Geçersiz optik akış ölçümü veya eksik metrik sıfır yerine çizgiyle gösterilir.

İşlem kullanıcı durdurduğunda, hata oluştuğunda ve normal tamamlandığında ayrı
durum mesajları gösterilir. Yeniden başlatma ve pencereyi kapatma, arka plan
işçisinin gerçekten çıkmasını bekler. Eşleştirme algoritması, veri yolları,
Kalman hesabı ve optik akış hesabı değiştirilmemiştir.

## Şablon eşitliği

Bu repo kökünde, sadece karşılaştırmak için:

```powershell
python -m mission_ui.sync 'C:\d_surucusu\simulasyon' --check
python -m mission_ui.sync 'C:\d_surucusu\tercom simulasyon' --check
```

Şablonu burada güncellediyseniz aynı komutları `--check` olmadan çalıştırarak
diğer repolara aktarın. Ters yönde eşitleme de desteklenir. Üç uygulama aynı
Python ortamına kurulacaksa şablon sürümlerini aynı tutun.

## Doğrulama

```powershell
python -m unittest discover -s tests -q
```

127 test: mevcut 122 test ve beş Qt entegrasyon testi. Qt testleri gerçek
pencere ve işçi sinyallerini kullanır; veri işleme çağrısını kontrollü bir
test işleviyle değiştirir. Gerçek model/raster setinin tamamını çalıştırmaz.
Pencereler 1480 × 920 ve 1180 × 760 boyutlarında render edilmiştir;
yerel önizlemeler `run_artifacts/ui-preview/` altındadır.
