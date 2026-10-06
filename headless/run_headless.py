#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_headless.py — Ekransiz (headless) Linux sunucu icin calistirma giris noktasi.

ONEMLI: Bu dosya ANA KODA DOKUNMAZ. Ayri bir klasorde (headless/) durur ve ana
isleme modulunu UST DIZINDEN import eder. Mevcut davranis birebir korunur.

Amac:
  - Hicbir GUI penceresi (cv2.imshow / namedWindow / waitKey / Qt) ACMAZ.
  - run_pipeline'a bos olmayan bir `callbacks` sozlugu gecer; ana hattaki tum
    pencere kodu zaten `if callbacks is None:` ile korundugundan TAMAMEN atlanir.
  - Per-frame ozet bilgileri stdout'a (ve istege bagli log dosyasina) yazar.
  - Istege bagli olarak her N karede bir overlay'li goruntuyu PNG diske kaydeder.

Kullanim ornekleri (bu klasorden veya ust dizinden calisabilir):
  python headless/run_headless.py
  python headless/run_headless.py --frame-every 10 --out-dir headless/headless_out
  python headless/run_headless.py --no-frames --log-file tm_run.log --log-level INFO
  python headless/run_headless.py --anlik guzergahlar/guzergah_2_tezde_ucus_1

Notlar:
  - Veri yollari (haritalar, model, guzergahlar, DEM) UST DIZINE gore cozulur;
    bu betigi nereye koyarsaniz koyun, ana proje kokunden calisir.
  - opencv-python yerine opencv-python-headless kurulu olabilir; bu betik hicbir
    GUI fonksiyonu cagirmadigindan headless paketle sorunsuz calisir.
  - cv2.imwrite GUI gerektirmez; PNG kaydi headless opencv ile de calisir.
"""

import os
import signal
import argparse
import importlib.util

import cv2

# Bu dosyanin klasoru = .../headless ; proje koku = bir ust dizin.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.environ.get("TM_PROJECT_ROOT", os.path.dirname(_HERE))
_CORE_NAME = "template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py"
_CORE_PY = os.path.join(_PROJECT_ROOT, _CORE_NAME)


def _load_core():
    """Ana isleme modulunu UST DIZINDEN yukler (if __name__=='__main__' blogu calismaz)."""
    if not os.path.isfile(_CORE_PY):
        raise FileNotFoundError(
            f"Ana isleme dosyasi bulunamadi: {_CORE_PY}\n"
            f"TM_PROJECT_ROOT ortam degiskeni ile proje kokunu belirtebilirsiniz."
        )
    spec = importlib.util.spec_from_file_location("tm_core", _CORE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Temiz kapatma icin global durdurma bayragi -----------------------------
_STOP = {"flag": False}


def _install_signal_handlers():
    def _handler(signum, _frame):
        _STOP["flag"] = True
        print(f"\n[headless] Sinyal alindi ({signum}); mevcut kare sonunda durulacak...",
              flush=True)
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _handler)
        except Exception:
            # Bazi platformlarda SIGTERM ayarlanamayabilir; sorun degil.
            pass


def _overlay_hud(frame, meta, max_width):
    """Kaydedilecek kareye meta'dan kisa bir bilgi paneli cizer.

    callbacks modunda ana hat metin HUD panelini cizmedigi icin (yalnizca
    scale-bar/pusula/ilerleme cubugu cizilir), burada ozeti biz ekliyoruz.
    Ardindan istege bagli olarak boyutu kuculterek kaydetmeye hazirlar.
    """
    img = frame
    try:
        lines = [
            f"FRAME: {meta.get('frame', '?')}/{meta.get('total', '?')}  ACC: {meta.get('acc', 0):.1f}%",
            f"HDG: {meta.get('hdg', 0):.1f} deg   ALT: {meta.get('alt', 0)} m",
            f"ERR: {meta.get('err_m', 0)} m   TM: {meta.get('max_val2', 0):.3f}",
            f"LAT: {meta.get('lat', 0):.6f}  LON: {meta.get('lon', 0):.6f}",
            f"SPD: {meta.get('spd_ms', 0):.2f} m/s ({meta.get('spd_kmh', 0):.1f} km/h)",
        ]
        h, w = img.shape[:2]
        # Yazi olcegini kare genisligine gore olcekle (kareler cok buyuk olabilir).
        fs = max(1.0, w / 900.0)
        th = max(2, int(fs * 2))
        sizes = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for s in lines]
        line_h = max(hh for (_, hh) in sizes)
        gap = int(line_h * 1.7)
        pad = int(line_h * 0.8)
        box_w = max(ww for (ww, _) in sizes) + 2 * pad
        box_h = gap * len(lines) + pad
        overlay = img.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        y = pad + gap
        for s in lines:
            cv2.putText(img, s, (pad + pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), th, cv2.LINE_AA)
            y += gap
    except Exception:
        pass

    # Kaydetmeden once boyut kucult (PNG boyutu/suresi icin).
    try:
        h, w = img.shape[:2]
        if max_width and w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    return img


def main():
    ap = argparse.ArgumentParser(
        description="Gorusel navigasyon template-matching hattini ekransiz (headless) calistirir.")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "headless_out"),
                    help="Kaydedilecek karelerin klasoru (varsayilan: headless/headless_out).")
    ap.add_argument("--frame-every", type=int, default=10,
                    help="Her N karede bir PNG kaydet (varsayilan: 10). 0/negatif = kaydetme.")
    ap.add_argument("--no-frames", action="store_true",
                    help="Hic kare kaydetme; yalnizca konsol/CSV ciktisi.")
    ap.add_argument("--frame-max-width", type=int, default=1600,
                    help="Kaydedilen karenin maks. genisligi (px). 0 = kucultme yok.")
    ap.add_argument("--log-file", default=None,
                    help="Belirtilirse ek logger bu dosyaya da yazar (LOG_TO_FILE).")
    ap.add_argument("--log-level", default=None,
                    help="DEBUG|INFO|WARNING|ERROR (varsayilan: RUN_CFG degeri).")
    # Sik degisen yollari komut satirindan ezmek icin (opsiyonel):
    ap.add_argument("--anlik", default=None, help="ANLIK_DIR'i ez (islenecek goruntu klasoru).")
    ap.add_argument("--model", default=None, help="MODEL_DIR'i ez.")
    ap.add_argument("--harita", default=None, help="HARITA_DIR'i ez.")
    ap.add_argument("--model-norm", default=None,
                    help="Anlik goruntu on isleme modu: HAM|HISTEQ|CLAHE|EDGE "
                         "(varsayilan: model sozlesmesi, yoksa RUN_CFG['MODEL_INPUT_NORM']).")
    ap.add_argument("--model-scale", default=None,
                    help="Model girdi olcegi: minus1_1|zero_1|raw|zscore "
                         "(varsayilan: model sozlesmesi, yoksa RUN_CFG['MODEL_INPUT_SCALE']). "
                         "Verilirse model on isleme sozlesmesi devre disi kalir.")
    args = ap.parse_args()

    tm = _load_core()

    # --- RUN_CFG ezmeleri (GUI ile alakali hicbir sey acmaz) ---------------
    # NOT: Bunlar yalnizca BU SURECTEKI RUN_CFG kopyasini ezer; kaynak dosya degismez.
    tm.RUN_CFG["DEBUG"] = False  # DEBUG pencereleri zaten callbacks ile atlanir; net olsun.
    if args.log_file:
        tm.RUN_CFG["LOG_TO_FILE"] = True
    if args.log_level:
        tm.RUN_CFG["LOG_LEVEL"] = args.log_level
    if args.anlik:
        tm.RUN_CFG["ANLIK_DIR"] = args.anlik
    if args.model:
        tm.RUN_CFG["MODEL_DIR"] = args.model
    if args.harita:
        tm.RUN_CFG["HARITA_DIR"] = args.harita
    if args.model_scale:
        # Elle verilen olcek sozlesmeyi ezer.
        scale = args.model_scale.strip().lower()
        if scale not in tm.MODEL_INPUT_SCALE_MODES:
            ap.error("--model-scale gecersiz: " + args.model_scale
                     + " (gecerli: " + ", ".join(tm.MODEL_INPUT_SCALE_MODES) + ")")
        tm.RUN_CFG["MODEL_INPUT_SCALE"] = scale
        tm.MODEL_INPUT_SCALE = scale
        tm.RUN_CFG["MODEL_INPUT_USE_SIDECAR"] = False

    if args.model_norm:
        # Modul sabitleri import aninda hesaplandigi icin ikisini birlikte ez.
        mode = args.model_norm.strip().upper()
        if mode not in tm.MODEL_INPUT_NORM_MODES:
            ap.error("--model-norm gecersiz: " + args.model_norm
                     + " (gecerli: " + ", ".join(tm.MODEL_INPUT_NORM_MODES) + ")")
        tm.RUN_CFG["MODEL_INPUT_NORM"] = mode
        tm.MODEL_INPUT_NORM = mode
        # Elle mod verildiginde model sozlesmesi devre disi kalir.
        tm.RUN_CFG["MODEL_INPUT_USE_SIDECAR"] = False

    tm.setup_logging()

    save_frames = (not args.no_frames) and (args.frame_every and args.frame_every > 0)
    if save_frames:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[headless] Kareler her {args.frame_every} adimda '{args.out_dir}' icine kaydedilecek.",
              flush=True)
    else:
        print("[headless] Kare kaydi KAPALI; yalnizca konsol/CSV ciktisi.", flush=True)

    _install_signal_handlers()

    def on_log(msg):
        print(msg, flush=True)

    def on_frame(frame, meta):
        if not save_frames:
            return
        try:
            fno = int(meta.get("frame", 0))
        except Exception:
            fno = 0
        if fno % args.frame_every != 0:
            return
        try:
            out = _overlay_hud(frame, meta, args.frame_max_width)
            path = os.path.join(args.out_dir, f"frame_{fno:06d}.png")
            cv2.imwrite(path, out)
        except Exception as e:
            print(f"[headless] kare kaydedilemedi ({fno}): {e}", flush=True)

    def stop_check():
        return _STOP["flag"]

    callbacks = {
        "on_log": on_log,
        "on_frame": on_frame,
        "stop_check": stop_check,
    }

    print(f"[headless] Proje koku: {_PROJECT_ROOT}", flush=True)
    print("[headless] run_pipeline baslatiliyor (GUI penceresi acilmayacak)...", flush=True)
    tm.run_pipeline(callbacks=callbacks)
    print("[headless] Bitti. Sonuc dosyalari (proje kokunde): "
          "sonuclar.csv / sonuclar.txt / modele_gore_sonuclar.txt",
          flush=True)


if __name__ == "__main__":
    main()
