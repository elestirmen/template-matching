# Unmanned Systems v15: devam notu (Claude Code için)

Bu dosya, claude.ai'de yürütülen çalışmanın Claude Code'a devridir. Önce bu dosyanın tamamını oku, sonra "Sıradaki iş" bölümünden başla.

## Kullanıcı ve çalışma kuralları
- Kullanıcı: Dr. Ahmet Ertuğrul Arık (Kapadokya Üniversitesi). Makale doktora tezinden; ikinci yazar Nuri Emrahoğlu (danışman).
- Kullanıcıya **doğal, açıklayıcı Türkçe** yaz. Yanıtlarda uzun tire (U+2014) kullanma.
- **Makalede yalnız klavye kısa tiresi "-"** kullanılır: U+2014, U+2013 ve U+2212 yasak. Bulursan cümleyi yeniden kurma, karakteri "-" ile değiştir.
- E-posta taslakları kısa olmalı.
- Özet en fazla 200 kelime (şu an 199).
- Sonuçlar dürüst raporlanır: test sonuçlarını gördükten sonra yapılan her yöntem değişikliği makalede öyle belirtilir. Uçuş tabloları yalnız kare bazlı kayıtlardan script ile üretilir; elle sayı yazılmaz.

## Arka plan
- Dergi: Unmanned Systems, makale US-D-26-00277, başlık "Cross-Autoencoder Structural Representations for GNSS-Denied UAV Localization by Template Matching...". Revizyon sonrası RET aldı.
- İki hakemin ana itirazları:
  - Uçuş tabloları iç tutarsız (RMSE ile yanlış sayısı Cauchy-Schwarz'a aykırı).
  - TP/FP/TN/FN ve F1 tanımları belirsiz; çözülemeyen kareler TN sayılmış.
  - Model seçimi ile değerlendirme ayrılmamış.
  - Replay her karede GNSS irtifası ve konumu kullanıyor, yani GNSS'siz çalışma gösterilmemiş.
  - Schleiss 2019 eksik, baseline'lar zayıf.
  - Düşük dokulu sorgular benchmark'tan elenmiş.
- Tutarsızlığın kaynağı bulundu:
  - Tez tabloları 70 m eşikle üretilmiş.
  - TN "üç kırpıntı kesişmedi" demekti.
  - RMSE tüm kareler üzerinden hesaplanmıştı.
  - Hata 300 m'yi geçince konum GPS ile geri çekiliyordu.
  - Tez tabloları makaleden çıkarıldı. Tüm uçuş sonuçları kare bazlı replay ile sıfırdan üretildi.

## Klasörler
- Kod ve veriler: `C:\d_surucusu\visual_navigation\`
  - `template-matching\flight_replay\`: uçuş replay altyapısı.
    - `replay_flights.py`: kare bazlı CSV üretir. Kendini en fazla 3 kez yeniden dener; `rerun_queue\` kuyruğu var.
    - `run_all.py` ve `run_replay.bat`: sürücü. Kaldığı yerden devam eder. Aynı anda iki kopya çalıştırma.
    - `make_tables.py`, `texture_probe.py`, `make_fig6.py`, `cae_infer.py`, `models_registry.py`.
    - Ana sonuçlar (eşitlemesiz protokol): `runs_ne\`, haritaları `maps_ne\`.
    - Eşitlemeli duyarlılık koşuları: `runs\`, haritaları `maps\`. Gri NCC koşuları da burada; gri temsil eşitlemeden etkilenmez.
    - Düşük doku testi: `texture_probe_ne\`.
    - `runs_ne\_yarim_f48_duyarlilik\`: yarım kalan ve kullanılmayan koşular.
  - `template-matching\manuscript_v15\`: makale üretim scriptleri. Bu dosya da burada.
  - `benchmark\`: geniş Ürgüp benchmark'ı.
    - `deney_araclari\sobel_benchmark.py` ve `.bat`.
    - `deney_araclari\recon_kontrol.py` ve `.bat`.
    - `deneyler\pipeline_20261108_222031\`: yeniden kurma deneyi.
  - `autoencoder_sat_to_map\egitim\autoencoder_unified.py`: yeniden kurma deneyi için `TARGET_MODE` eklendi (varsayılan "map"). Git'te değişmiş görünür, commit edilmedi.
- Makale dosyaları: `C:\Users\ertugrul\OneDrive - kapadokya.edu.tr\Arastirma\Tezden_Makale\Unmanned_Systems\`
  - `Arik_Emrahoglu_UnmannedSystems.docx`: gönderilen v14. Üretimin kaynağı budur.
  - `v15\`: son hali.
    - `Arik_Emrahoglu_UnmannedSystems_v15.docx` ve `.pdf`.
    - `Arik_Emrahoglu_UnmannedSystems_v15_Hakem_Yaniti.docx`.
    - Eski `_taslak` dosyaları.
  - OneDrive klasörüne Claude Code'da `/add-dir` ile erişim ver.
- Python ortamı: `%USERPROFILE%\anaconda3\envs\visual_navigation_cuda` (TF 2.10 GPU). Makale scriptleri için bu ortama `python-docx`, `pandas`, `pillow` ve `tabulate` gerekir; eksikse `pip install`. PDF için LibreOffice (`soffice --headless --convert-to pdf`) ya da Word.

## v15'i yeniden üretmek
`template-matching\manuscript_v15\` içinde:
```
python make_v15.py out\Arik_Emrahoglu_UnmannedSystems_v15.docx ^
  --runs ..\flight_replay\runs_ne ..\flight_replay\runs@gray_ --eq1 ..\flight_replay\runs ^
  --texture ..\flight_replay\texture_probe_ne\texture_probe_summary.csv ^
  --recon inputs\recon.json --fig6 inputs\fig6_v15.png --work out
python make_response.py out\numbers.json out\Arik_Emrahoglu_UnmannedSystems_v15_Hakem_Yaniti.docx
```
- `--src` verilmezse v14 yolu `build_v15.py` içindeki OneDrive yoludur.
- Zincir:
  - `make_tables.py`: ortak uygun kare kümesi `gray_oracle`'dan alınır; tablolar `out\tables`.
  - `fill_extra.py`: sayılar, kamera ofseti, eşitleme farkları; çıktı `out\numbers.json`.
  - `narrative.py`: sonuca bağlı paragraflar. Karşılaştırma ifadeleri sayılardan türetilir.
  - `build_v15.py`: yöntem bölümleri ve kaynaklar.
  - `build_results.py`: Tablo 3-5, Bölüm 4.1-4.2 ve Şekil 6.
  - `make_v15.py` sonundaki `FIXES`: v14'ten kalan cümlelerdeki düzeltmeler.
- Kontroller:
  - `out\tables\by_group.csv` ve `by_route.csv` dosyalarında `consistent` sütunu tümüyle True olmalı.
  - Doktora rotalarında her satırda eligible = 1.976 olmalı.
  - Metinde "?", "PLACEHOLDER" ve yasak tire olmamalı.
  - Özet en fazla 200 kelime.
- Şekil 6 paneli değişirse (template-matching kökünden):
  - `python flight_replay\make_fig6.py flight_replay\runs_ne\m4e300_track_causal.csv Urban5 urgup_30_cm_yeni_gmaps_utm.tif flight_replay\fig6_v15\a_urban5.png --title "(a) Urban 5"`
  - Aynısını `Rural1` ve `karlik_30_cm_bingmap_utm.tif` ile tekrarla.
  - Sonra `python manuscript_v15\compose_fig6.py a.png b.png manuscript_v15\inputs\fig6_v15.png`.

## Ana sonuçlar
Eşitlemesiz protokol, 11 doktora rotası, 1.976 uygun kare, 30 m eşik.

| Temsil | Oracle (GPS merkezli 3×3 km) | Nedensel (GNSS yalnız ilk karede) | GNSS destekli |
|---|---|---|---|
| Gri NCC | 67,3 | 71,4 | 76,7 |
| Gradyan büyüklüğü (Sobel, σ1,5) | 70,9 | 73,8 | - |
| Model 4 hattı (f48-k5-s2, epoch 300; uçurulan epoch 250 kayıp) | 76,4 | 74,8 | 82,8 |
| f48-k4-s1 ckpt 1850 (benchmark seçimi, Tablo 6) | 68,6 | 70,5 | 78,7 |
| f48-k4-s1 ckpt 1550 (çekirdek mozaikte en iyi) | 73,2 | 80,5 | - |

- **Ek uçuşlar** (Urban 9-11, Rural 4-5; katı hold-out değil, "additional flights"): Sobel tüm checkpoint'lere eşit ya da üstün (oracle 76,2). Rural 4'te uygun kare yok; Rural 5'in 106 karesi DEM dışında.
- **Kamera ofseti** (gövde ekseni, ileri/sağ):
  - Mavic 2 Pro yaklaşık -6/-7 m, Mavic 2 Zoom yaklaşık +5/-22 m.
  - Diğer rotalardan tahmin edilip düzeltildiğinde Model 4 nedensel hassasiyeti 93,8'den 99,1'e çıkıyor, medyan hata 14,5 m'den 6,5 m'ye iniyor.
  - Tablolar düzeltilmemiş değerleri veriyor.
- **Kırsal kayıp mekanizması:**
  1. Eşleşmeyen kareler sırasında tahmini konum kayar.
  2. Yanlış konumun DEM yüksekliği okunur.
  3. Kare yaklaşık %20 küçük ölçeklenir.
  4. Takip toparlanamaz (Rural 1-2).
- **Duyarlılık** (Model 4, Tablo 5):

  | Bozulma | Başarı (%) |
  |---|---|
  | Yön +5° / -5° | 65,9 / 55,3 |
  | Yön +10° / -10° | 20,6 / 15,9 |
  | Yükseklik +%5 / -%5 | 73,6 / 73,5 |
  | Yükseklik +%10 / -%10 | 55,2 / 49,5 |
  | Başlangıç 200 m | 74,7 |
  | Başlangıç 500 m | 67,1 |

- **Eşitleme:** Eğitim betikleri histogram eşitlemesi yapıyordu; uçuş kodu ve benchmark yapmıyor. Ana protokol eşitlemesiz. Eşitlemeli koşunun farkı:

  | | Oracle | Nedensel |
  |---|---|---|
  | Model 4 | -2,1 | +1,5 |
  | ckpt 1850 | -9,4 | -7,2 |

- **Yeniden kurma kontrolü** (1000 sorgu, global, yön × varyant ortalaması):
  - Girdiyi yeniden kuran ağ %31,7.
  - Gri NCC %46,1.
  - Harita hedefli ağ %56,9 (aynı tarif, eylül koşusu `pipeline_20260924_065751`).
  - Tekrar eğitimi (`tekrar2`) GPU kilitlenmesiyle yarım kaldı; gerekli değil.
- **Düşük doku testi** (±1 km pencere, temiz karolar):
  - Düzgün dokulu karolarda gri 68,4, Sobel 91,8, CAE (ckpt 1850) 85,3.
  - Düşük dokulu karolarda gri 77,7, Sobel 90,2, CAE 88,0.
  - Eleme oranları: Google %1,8, Bing %8,3, ayrıca %3,5-3,9 eksik veri.
- **Benchmark (Tablo 6):**
  - Gri NCC %49,1 / %46,2, CAE ckpt 1850 %69,0 / %62,7 (temiz / hard_v1, global, 4000 sorgu).
  - Mekânsal ayrım (kuzey/güney yarısı, 1 km tampon) kazancı koruyor: +20,8 / +16,9 ve +19,3 / +14,7 puan.

## Sobel benchmark sonucu ve v16 (9 Kasım 2026, Claude Code)
- Koşu 10:13-11:52 arasında hatasız bitti. Çıktı: `benchmark\outputs\sobel_baseline_4000\` (`results.jsonl`, `benchmark_results.xlsx`,
  `sobel_summary.json`, `table6_global_raw_1850.csv` = Tablo 6 koşusundan süzülen RAW ve ckpt 1850 kayıtları).
- `python deney_araclari\sobel_benchmark.py summary` (benchmark klasöründen, Anaconda base python yeterli) özeti üretir:
  gri koşu Tablo 6'yı 16.000 sorguda birebir tekrarladı; Sobel 78,6 / 77,4 (temiz / hard_v1, iki yön ortalaması),
  ckpt 1850 69,0 / 62,7, gri 49,1 / 46,2. Eşleştirilmiş fark +9,6 [8,1; 11,0] ve +14,7 [13,2; 16,3] puan (1 km blok bootstrap).
  Sobel iki yönde, kuzey/güney yarılarında ve bütün entropi dilimlerinde önde; 11 sorgu düşük şablon varyansıyla reddedildi (başarısız sayıldı).
- **Karar (kullanıcı ve Nuri Emrahoğlu, 9 Kasım):** makale "A çerçevesi" ile yeniden kuruldu: öğrenilmiş ve klasik yapısal
  temsilin kontrollü karşılaştırması. Başlık, anahtar kelimeler, özet, giriş sorusu ve katkı maddesi, 3.8'e Sobel sarmalayıcı
  paragrafı, 4.3'e Sobel paragrafı, asimetri paragrafı, Tablo 7'ye satır ve altyazı, Şekil 7 altyazısı, düşük doku cümlesi,
  tartışma (yorum, literatür, geçerlilik vi, gelecek iş) ve sonuç bölümü değişti. Hepsi betikten üretilir:
  `fill_extra.py --sobel-bench` sayıları `sb_*` anahtarlarıyla `numbers.json`'a yazar; `narrative.py` metni bunlardan kurar
  ve Sobel CAE'nin üstünde değilse bilerek durur (`SystemExit`).
- Üretim (manuscript_v15 klasöründen, Anaconda **base** python; conda env'de python-docx yok; konsol için `PYTHONUTF8=1` gerekli):
  ```
  set PYTHONUTF8=1
  python make_v15.py out\Arik_Emrahoglu_UnmannedSystems_v16.docx ^
    --runs ..\flight_replay\runs_ne ..\flight_replay\runs@gray_ --eq1 ..\flight_replay\runs ^
    --texture ..\flight_replay\texture_probe_ne\texture_probe_summary.csv ^
    --recon inputs\recon.json --fig6 inputs\fig6_v15.png ^
    --sobel-bench ..\..\benchmark\outputs\sobel_baseline_4000\sobel_summary.json --work out
  python check_v16.py out\Arik_Emrahoglu_UnmannedSystems_v16.docx out\tables --show
  python make_response.py out\numbers.json out\Arik_Emrahoglu_UnmannedSystems_v16_Hakem_Yaniti.docx
  python to_pdf.py out\Arik_Emrahoglu_UnmannedSystems_v16.docx
  ```
- v16 çıktıları kullanıcının onayıyla OneDrive'a kopyalandı: `Unmanned_Systems\v16\` (docx, pdf, hakem yanıtı docx/pdf,
  `numbers_v16.json`). `00_Gonderim_Notlari.md`'ye v16 bölümü eklendi. Kaynak hâlâ `manuscript_v15\out\`; yeniden üretince
  OneDrive kopyasını da yenile.

## IJEG v17 (9 Kasım 2026)
- Hedef dergi IJEG oldu. v16 docx, `ijeg\build_ijeg.py` ile iki sütunlu IJEG şablonuna dökülür:
  `python ijeg\build_ijeg.py out\..._v16.docx "<IJEG>\IJEG_Template.docx" ijeg\refs_apa_v16.txt out\..._v17_IJEG.docx "<IJEG>\..._v09_IJEG_icerik.docx"`
  (son argüman APA kaynakçadaki italikler için). Çıktı OneDrive `Tezden_Makale\IJEG\` klasörüne kopyalanır.
- Şekil 1, 7, 8, 9: `ijeg\make_figs.py` benchmark çıktılarından (`run_20260809_225158`, `urgup_cevresi_models_4000`,
  `urgup_bes_nokta_global_100`, `margin_gate`, `epoch_sweep_2000`) 18 cm / ≥437 dpi PNG üretir (`ijeg\figs\`); `build_ijeg.py`
  FIG_OVERRIDE bunları v17'ye koyar. v16 (WS biçimi) hâlâ eski 300 dpi PNG'leri taşır.
- Kapak mektubu: `ijeg\make_cover_letter.py` → `IJEG\Arik_Emrahoglu_IJEG_Cover_Letter.docx` (sayılar out\numbers.json'dan).
- Sürüm ve dergi notları: OneDrive `Tezden_Makale\_Ortak\00_Surum_ve_Dergi_Rehberi.md` ("IJEG v17 sonrası kalan işler").

## IJEG v18 (9 Kasım 2026, claude.ai)
- Yeni: `flight_ci.py` (uçuş farklarının eşleştirilmiş dairesel blok bootstrap aralıkları; rota içinde katmanlı, 25 kare, 10.000 tekrar; 10/50 kare ve rota bazlı duyarlılık). `make_v15.py` bunu `out\tables\flight_ci.csv/.json` olarak üretir ve `fill_extra.py --flight-ci` ile `fd_*` ve `cov_*` anahtarlarını `numbers.json`'a yazar.
- `narrative.py`: `ci_oracle` (4.1 sonu), `ci_causal` (4.2), `coverage` ve `ci_method` (3.7 Outcomes paragrafı), tartışma (rota bazlı kontrol dahil) ve sonuç cümleleri sayılardan kurulur; aralık, ifade edilen farkın yönüne göre verilir.
- `build_ijeg.py`: üstbilgi tüm sayfalarda "2026, 11(x), 00-0X".
- Üretim (Linux/VM'de denendi; Windows'ta aynı komutlar `set PYTHONUTF8=1` ile):
  ```
  python make_v15.py out\v18\Arik_Emrahoglu_v18_WS_kaynak.docx --runs ..\flight_replay\runs_ne ..\flight_replay\runs@gray_ ^
    --eq1 ..\flight_replay\runs --texture ..\flight_replay\texture_probe_ne\texture_probe_summary.csv --recon inputs\recon.json ^
    --fig6 inputs\fig6_v15.png --sobel-bench ..\..\benchmark\outputs\sobel_baseline_4000\sobel_summary.json --work out
  python ijeg\build_ijeg.py out\v18\Arik_Emrahoglu_v18_WS_kaynak.docx "<IJEG>\IJEG_Template.docx" ijeg\refs_apa_v16.txt ^
    out\v18\Arik_Emrahoglu_StructuralRepresentations_UAV_Localization_v18_IJEG.docx "<IJEG>\..._v09_IJEG_icerik.docx"
  python to_pdf.py out\v18\Arik_Emrahoglu_StructuralRepresentations_UAV_Localization_v18_IJEG.docx
  ```
- v18 docx OneDrive `IJEG\` klasörüne kopyalandı; Word PDF'i henüz alınmadı (`out\v18\*_libreoffice_kontrol.pdf` yalnız kontrol içindir).
- Değişmeden kalanlar: özet (198 kelime), kapak mektubu, tablolar ve şekiller. `check_v16.py` temiz (tire yok, tutarlılık True).

### v18 son kontrol düzeltmeleri (10 Kasım 2026, claude.ai)
- Word PDF'inde iki negatif sayı satır sonunda bölünüyordu ("(-" / "0.8 to 5.7)" ve "+1.5 and -" / "7.2 points"). `build_ijeg.py` artık gövde metnindeki negatif sayıların işaretini Word'ün bölünmez tiresi (`w:noBreakHyphen`, Word'de Ctrl+Shift+-; görünüşü klavye tiresiyle aynı) olarak yazar (`add_text`, 24 yer). Diğer bütün tireler düz "-" kalır. Kullanıcı istemezse `add_text` içindeki dönüşümü kaldırmak yeterli.
- Yazar satırına şablondaki ORCID simgeleri geri kondu (`orcid_icon`; Arık 0000-0002-7952-4311, Emrahoğlu 0000-0003-4347-5279).
- Data availability: uçuşlarda koşulan üç checkpoint benchmark deposunda zaten herkese açık (SHA-256 ile bayt bayt aynı; `flight_replay\models_registry.py` içinde `PUBLIC`). Cümle buna göre düzeltildi; uçuş görüntüleri hâlâ "istek üzerine" (`text_final.py` DATA_AVAIL).
- Düzeltilmiş docx `out\v18\` içinde. OneDrive `IJEG\` klasöründeki v18 docx/pdf çifti henüz eski; Word PDF'i alınınca ikisi birlikte değiştirilecek. PDF alındıktan sonra kontrol: 1. sayfada iki ORCID simgesi, 11. sayfada "(-0.8 to 5.7)" tek satırda ve eksi işareti normal görünüyor mu.
- Açık (kullanıcının kararı): autoencoder_sat_to_map'te yeniden kurma kontrolünün kullandığı `TARGET_MODE` değişikliği (`egitim\autoencoder_unified.py`) commit edilmedi ve yerel dal main'e gönderilmedi; Zenodo DOI.

### v18 dil kontrolü (10 Kasım 2026, claude.ai)
- Metnin tamamı okundu. İlk gönderimde yeri olmayan revizyon dili kaldırıldı: "added as a control in this revision" -> "It serves as a control and was run..."; "Thresholded F1 scores are no longer used" -> "The thresholded F1 scores of the doctoral study are not used" (ikisi de `build_ijeg.py` TEXT_FIXES; WS kaynağı değişmedi).
- Bölüm 2.5'te düz metin olarak kalan "S_(w)" ve "I_(w)" alt simge yapıldı (`docx_tools.set_par` artık `_{w}` işaretini tanır; `text_methods.py`).
- Tekrar eden yüklemler ve okunması zor yerler: "leads ... and leads / trails ... and trails" (`narrative.py` `vs2`), "checkpoint 1850 70.5%" sıralaması (`build_results.py`), kamera ofseti parantezi (`narrative.py`), mimari cümlesi (`text_bench.py`).
- Yazım birliği: "pre-process" -> "preprocess", "Data Availability)" -> "Data availability)", düz kesme -> kıvrık kesme.
- Kapak mektubu artık üretildiği günün tarihini taşır (`make_cover_letter.py`); gönderim günü yeniden üret.
- Düzeltilmiş docx yine `out\v18\` içinde; Word PDF'i alınıp IJEG klasörüne birlikte kopyalanacak.
- Açık (kullanıcının kararı): Acknowledgement'taki yapay zekâ kullanım beyanı v14'ten geliyor ("Claude Opus 5 ... software code and English-language editing"); kapsamı ve model adı güncel durumu tam yansıtmıyor olabilir.

## Sıradaki iş
1. **GitHub:** 9 Kasım'da iki depo main'e birleştirilip push edildi (template-matching 7fbc6d4, benchmark 8724fe9). Kalan: template-matching ve autoencoder_sat_to_map'te lisans ve atıf dosyaları .gitignore yüzünden yüklenmedi; üç deponun CITATION.cff/.zenodo.json'u eski başlık ve "Drones" diyor, IJEG'e göre güncellenmeli. Bu DEVAM.md değişikliği henüz commit edilmedi.
2. **Kullanıcının kararını bekleyen konular:**
   - Kamera ofseti düzeltilmiş sonuç tabloda ek satır olsun mu, yalnız metinde mi kalsın?
   - Kırsal yer yüksekliği düzeltmesi (konum belirsizken son güvenilir konumun yer yüksekliği): önerimiz gelecek çalışma, çünkü sonradan yöntem değişikliği olur.
4. **Gönderim öncesi:**
   - Şekil 6 değişti: `Unmanned_Systems\Figures\CMYK_TIFF_300dpi` için yeni 300 dpi CMYK TIFF üret (`Figures\README.txt` yöntemine bak). Altyazıdaki Maxar notu `narrative.py` içinde eklendi; v15'i yeniden üretince docx'e girer.
   - `Unmanned_Systems\00_Gonderim_Notlari.md` gönderim notlarıdır; v15 bölümü eklendi, gönderim paketi tablosunu v15 dosyalarıyla güncelle.
   - `flight_replay` kodu ve kare bazlı kayıtlar GitHub'a (elestirmen/template-matching) konmalı; Data Availability bunu varsayıyor.
   - Kapak mektubu güncellenmeli.
   - Bağımsız denetimin "kısmen" bıraktığı noktalar yanıt mektubunda açıkça yazılı. Ek eylem yalnız Sobel için gerekiyor.

## Bağımsız denetimin kalan notları (v15 sonrası)
Çözülenler: tutarsız istatistikler, tanımlar ve paydalar, TN sayımı, yeniden hesaplama.

Kısmen kalanlar ve nedenleri:
- Takip parametreleri Urban 1 ve Urban 5'te ayarlandı. Bu metinde yazılı.
- ckpt 1850 aynı bölgenin uydu görüntüsünde seçildi; sonuçlar in-domain.
- Uçurulan epoch-250 dosyası kayıp. Epoch 300 vekili benchmark'ta koşmadı; benchmark'taki girdiler epoch 450 ve 500.
- Düşük doku testi yalnız ±1 km pencerede.
- Sobel global benchmark sonucu bekleniyor.

## Altyapı uyarıları
- **GPU sorunları:** RTX 5060 Ti, CUDA 11.2 ile PTX JIT çalışıyor. Ara sıra `CUDA_ERROR_ILLEGAL_ADDRESS`, `0xC0000409` hataları ve cuDNN "redzone" uyarıları çıkıyor. 9 Kasım 07:21'de GPU tamamen kilitlendi; yeniden başlatma gerekti.
- **Supervisor ve kuyruk:** `replay_flights.py` supervisor'ı çöken işi yeniden dener. Yarım kalan rota için `flight_replay\rerun_queue\<out yolu, / yerine _>.txt` dosyasına bir JSON argüman listesi yazılabilir.
- **CUDA önbelleği:** `CUDA_CACHE_PATH = benchmark\outputs\cuda_cache`, en fazla 4 GB. Önbellek olmadan her süreç yaklaşık 4,5 dakika ısınır.
- **Paralel çalıştırma:** İki GPU işini aynı anda çalıştırmak çöküş riskini artırır. Sobel benchmark çalışırken başka GPU işi başlatma.
