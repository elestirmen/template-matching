"""
konum_ui_qt.py  —  PyQt5 tabanli navigasyon goruntuleyicisi.

Mimarisi:
  - ProcessingWorker (QThread): tum image-processing + ML islemlerini arka planda calistirir.
  - MapDisplay       (QLabel) : her frame'i numpy -> QPixmap olarak gosterir.
  - ControlPanel     (QWidget): toggle'lar, canli metrik kartlari, log ciktisi.
  - MainWindow       (QMainWindow): QSplitter ile sol (harita) + sag (kontrol) duzeni.

Kullanim:
    python konum_ui_qt.py
"""

import sys
import os
import threading
import math
import time
import importlib.util

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QSplitter, QFrame, QCheckBox, QPushButton, QProgressBar,
    QPlainTextEdit, QScrollArea, QSizePolicy, QStatusBar, QGridLayout,
    QGroupBox, QSpacerItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor


# ---------------------------------------------------------------------------
# Ana islem modulunu yukle (if __name__=='__main__' blogu calistirilmaz)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE_PY = os.path.join(_HERE,
    "template_matching_parallel_processing_560_hizli_solust_sagalt_koordinat_fonksiyonlar_icinde_cursor.py")

_spec = importlib.util.spec_from_file_location("tm_core", _CORE_PY)
_tm   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)

# Kisayollar – sadece modul-duzeyi fonksiyonlar
RUN_CFG                  = _tm.RUN_CFG
log_cuda_info_once       = _tm.log_cuda_info_once
load_model_compat        = _tm.load_model_compat
_normalize_ext_set       = _tm._normalize_ext_set
_list_files_filtered     = _tm._list_files_filtered
_filter_candidates       = _tm._filter_candidates
make_rc_to_ll            = _tm.make_rc_to_ll
parse_exif               = _tm.parse_exif
piksel_bul_fast          = _tm.piksel_bul_fast
find_corner_coordinates  = _tm.find_corner_coordinates
rotate_image             = _tm.rotate_image
largest_rotated_rect     = _tm.largest_rotated_rect
crop_around_center       = _tm.crop_around_center
cuda_resize_if_available = _tm.cuda_resize_if_available
is_valid_slice           = _tm.is_valid_slice
match_three              = _tm.match_three
intersection             = _tm.intersection
haversine_distance       = _tm.haversine_distance
draw_plane_icon_v2       = _tm.draw_plane_icon_v2
draw_scale_bar           = _tm.draw_scale_bar
draw_compass             = _tm.draw_compass
rmse                     = _tm.rmse
mae                      = _tm.mae
standart_sapma           = _tm.standart_sapma
dosyaya_yaz              = _tm.dosyaya_yaz

PATCH_SIZE             = _tm.PATCH_SIZE
PATCH_HALF             = _tm.PATCH_HALF
PRED_BORDER            = _tm.PRED_BORDER
USE_PYRAMID            = _tm.USE_PYRAMID
COARSE_SCALE           = _tm.COARSE_SCALE
ROI_PAD_FACTOR         = _tm.ROI_PAD_FACTOR
DRAW_TRAJECTORY        = _tm.DRAW_TRAJECTORY
TRAJECTORY_DRAW_POINTS = _tm.TRAJECTORY_DRAW_POINTS
TRAJECTORY_MAX_POINTS  = _tm.TRAJECTORY_MAX_POINTS
TRAJECTORY_LINE_THICKNESS = _tm.TRAJECTORY_LINE_THICKNESS
TRAJECTORY_POINT_RADIUS   = _tm.TRAJECTORY_POINT_RADIUS
RAKIM_DUZELTME         = _tm.RAKIM_DUZELTME
BASARI_ESIGI_KM        = _tm.BASARI_ESIGI_KM
CERCEVE_BOYUTU_MAX     = _tm.CERCEVE_BOYUTU_MAX
FARK_MAX               = _tm.FARK_MAX
CAMERA_SENSOR_BY_MODEL = _tm.CAMERA_SENSOR_BY_MODEL
benchmark              = _tm.benchmark
dirname                = _tm.dirname


# ---------------------------------------------------------------------------
# Renk paleti (Qt stylesheet'te kullanilacak)
# ---------------------------------------------------------------------------
C_BG       = "#1a1b2e"
C_PANEL    = "#1e2030"
C_BORDER   = "#3a3d55"
C_ACCENT   = "#4285f4"
C_SUCCESS  = "#4caf50"
C_WARN     = "#ff9800"
C_TEXT     = "#e8eaf0"
C_MUTED    = "#7b7f9e"
C_CARD_BG  = "#252740"
C_ON       = "#4caf50"
C_OFF      = "#3a3d55"


# ---------------------------------------------------------------------------
# ProcessingWorker
# ---------------------------------------------------------------------------
class ProcessingWorker(QThread):
    frame_ready     = pyqtSignal(object)          # numpy BGR array
    crop_ready      = pyqtSignal(object, object)  # (crop_bgr, model_gray)
    metrics_update  = pyqtSignal(dict)            # metrik sozlugu
    log_signal      = pyqtSignal(str)             # konsol satiri
    finished_signal = pyqtSignal(dict)            # son metrikler

    def __init__(self):
        super().__init__()
        self._running = True
        self._mutex   = QMutex()
        # UI gorunum/islev durumu (Qt butonlari tarafindan degistiriliyor)
        self._ui_state = {
            "trajectory":   bool(DRAW_TRAJECTORY),
            "inner_frame":  bool(RUN_CFG.get("SHOW_INNER_FRAME", False)),
            "roi_frame":    bool(RUN_CFG.get("SHOW_ROI_FRAME", True)),
            "tm_boxes":     bool(RUN_CFG.get("SHOW_TM_BOXES", True)),
            "kalman":       bool(_tm.USE_KALMAN),  # Kalman filtresi calisma-aninda ac/kapa
        }

    def set_ui_state(self, key: str, value: bool):
        with QMutexLocker(self._mutex):
            self._ui_state[key] = value

    def get_ui_state(self, key: str, default=False) -> bool:
        with QMutexLocker(self._mutex):
            return self._ui_state.get(key, default)

    def stop(self):
        self._running = False

    def _log(self, msg: str):
        self.log_signal.emit(str(msg))

    # ------------------------------------------------------------------
    def run(self):
        def _on_frame(vis_bgr, metadata):
            self.frame_ready.emit(vis_bgr)
            vc = metadata.pop('vis_crop',  None)
            vm = metadata.pop('vis_model', None)
            self.metrics_update.emit(metadata)
            if vc is not None and vm is not None:
                self.crop_ready.emit(vc, vm)

        def _on_log(msg):
            self.log_signal.emit(str(msg))

        def _stop_check():
            return not self._running

        def _ui_state_update():
            with QMutexLocker(self._mutex):
                return dict(self._ui_state)

        callbacks = {
            'on_frame':        _on_frame,
            'on_log':          _on_log,
            'stop_check':      _stop_check,
            'ui_state_update': _ui_state_update,
        }
        try:
            _tm.run_pipeline(callbacks)
        except Exception as exc:
            self.log_signal.emit(f"HATA: {exc}")
        self.finished_signal.emit({"tamamlandi": True})


# ---------------------------------------------------------------------------
# MapDisplay — numpy BGR -> QPixmap
# ---------------------------------------------------------------------------
class MapDisplay(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(700, 700)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {C_BG}; border-radius: 6px;")
        self.setText("Baslatilmasi bekleniyor...")
        self.setFont(QFont("Segoe UI", 14))
        self.setStyleSheet(f"background:{C_BG}; color:{C_MUTED}; border-radius:6px;")

    def update_frame(self, arr: np.ndarray):
        try:
            h, w = arr.shape[:2]
            rgb  = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            qi   = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qi).scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(pix)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SmallImagePanel — baslikli kucuk goruntu paneli
# ---------------------------------------------------------------------------
class SmallImagePanel(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.setStyleSheet(f"background:{C_BG}; border:1px solid {C_BORDER}; border-radius:6px;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        lbl_title.setStyleSheet(f"color:{C_MUTED}; background:transparent; border:none;")
        layout.addWidget(lbl_title)

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img.setStyleSheet("background:transparent; border:none;")
        layout.addWidget(self._img)

    def update_frame(self, arr):
        try:
            if arr is None:
                return
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            h, w = arr.shape[:2]
            rgb  = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            qi   = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
            pix  = QPixmap.fromImage(qi).scaled(
                self._img.width() or 300, self._img.height() or 200,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._img.setPixmap(pix)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# MetricCard — tek bir metrik gosteren kart
# ---------------------------------------------------------------------------
class MetricCard(QFrame):
    def __init__(self, label: str, unit: str = "", wide: bool = False):
        super().__init__()
        self._unit = unit
        self.setStyleSheet(f"""
            QFrame {{
                background: {C_CARD_BG};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
                padding: 2px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        lbl = QLabel(label.upper())
        lbl.setFont(QFont("Segoe UI", 9, QFont.Normal))
        lbl.setStyleSheet(f"color:{C_MUTED}; background:transparent; border:none;")

        self._val = QLabel("--")
        fs = 16 if not wide else 13
        self._val.setFont(QFont("Consolas", fs, QFont.Bold))
        self._val.setStyleSheet(f"color:{C_TEXT}; background:transparent; border:none;")

        layout.addWidget(lbl)
        layout.addWidget(self._val)

    def set_value(self, v):
        try:
            self._val.setText(f"{v}{self._unit}")
        except Exception:
            self._val.setText("--")

    def set_color(self, color: str):
        self._val.setStyleSheet(f"color:{color}; background:transparent; border:none;")


# ---------------------------------------------------------------------------
# ControlPanel
# ---------------------------------------------------------------------------
class ControlPanel(QWidget):
    toggle_changed = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(280)
        self.setMaximumWidth(360)
        self.setStyleSheet(f"background:{C_PANEL};")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Baslik
        title = QLabel("NAVIGASYON")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet(f"color:{C_ACCENT}; letter-spacing:2px;")
        root.addWidget(title)
        root.addWidget(self._separator())

        # Start / Stop butonlari
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Baslat")
        self.btn_stop  = QPushButton("Durdur")
        for btn, col in [(self.btn_start, C_SUCCESS), (self.btn_stop, C_WARN)]:
            btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
            btn.setFixedHeight(36)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background:{col}; color:#fff;
                    border-radius:6px; border:none;
                }}
                QPushButton:hover {{ background:{col}dd; }}
                QPushButton:disabled {{ background:{C_BORDER}; color:{C_MUTED}; }}
            """)
            btn_row.addWidget(btn)
        self.btn_stop.setEnabled(False)
        root.addLayout(btn_row)
        root.addWidget(self._separator())

        # Metrik kartlari (2 sutun)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.c_hdg  = MetricCard("HDG",   "°")
        self.c_alt  = MetricCard("ALT",   " m")
        self.c_err  = MetricCard("ERR",   " m")
        self.c_spd  = MetricCard("SPD",   " km/h")
        self.c_lat  = MetricCard("LAT",   "",  wide=True)
        self.c_lon  = MetricCard("LON",   "",  wide=True)
        self.c_acc  = MetricCard("ACC",   "%")
        self.c_tm   = MetricCard("TM",    "")
        for idx, card in enumerate([self.c_hdg, self.c_alt, self.c_err, self.c_spd,
                                    self.c_acc, self.c_tm]):
            grid.addWidget(card, idx // 2, idx % 2)
        root.addLayout(grid)

        # LAT / LON tam genislik
        for card in (self.c_lat, self.c_lon):
            root.addWidget(card)

        root.addWidget(self._separator())

        # Ilerleme
        prog_lbl = QLabel("ILERLEME")
        prog_lbl.setFont(QFont("Segoe UI", 9))
        prog_lbl.setStyleSheet(f"color:{C_MUTED};")
        root.addWidget(prog_lbl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(22)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background:{C_BG}; border:1px solid {C_BORDER};
                border-radius:4px; color:{C_TEXT}; font-size:11px;
            }}
            QProgressBar::chunk {{
                background:{C_ACCENT}; border-radius:3px;
            }}
        """)
        root.addWidget(self.progress)
        root.addWidget(self._separator())

        # Toggle'lar
        toggle_lbl = QLabel("GORUNUM / FILTRE")
        toggle_lbl.setFont(QFont("Segoe UI", 9))
        toggle_lbl.setStyleSheet(f"color:{C_MUTED};")
        root.addWidget(toggle_lbl)

        self._toggles: dict[str, QCheckBox] = {}
        toggle_defs = [
            ("kalman",      "Kalman Filtresi", bool(_tm.USE_KALMAN)),
            ("trajectory",  "Trajektori",  bool(_tm.DRAW_TRAJECTORY)),
            ("roi_frame",   "ROI Cerceve", bool(_tm.RUN_CFG.get("SHOW_ROI_FRAME", True))),
            ("tm_boxes",    "TM Kutulari", bool(_tm.RUN_CFG.get("SHOW_TM_BOXES", True))),
            ("inner_frame", "Ic Cerceve",  bool(_tm.RUN_CFG.get("SHOW_INNER_FRAME", False))),
        ]
        for key, label, default in toggle_defs:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.setFont(QFont("Segoe UI", 10))
            cb.setStyleSheet(f"""
                QCheckBox {{ color:{C_TEXT}; spacing:8px; }}
                QCheckBox::indicator {{
                    width:18px; height:18px; border-radius:4px;
                    border:1px solid {C_BORDER}; background:{C_BG};
                }}
                QCheckBox::indicator:checked {{
                    background:{C_ACCENT}; border:1px solid {C_ACCENT};
                }}
            """)
            cb.stateChanged.connect(lambda state, k=key: self.toggle_changed.emit(k, bool(state)))
            root.addWidget(cb)
            self._toggles[key] = cb

        root.addWidget(self._separator())

        # Log alani
        log_lbl = QLabel("LOG")
        log_lbl.setFont(QFont("Segoe UI", 9))
        log_lbl.setStyleSheet(f"color:{C_MUTED};")
        root.addWidget(log_lbl)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setStyleSheet(f"""
            QPlainTextEdit {{
                background:{C_BG}; color:{C_MUTED};
                border:1px solid {C_BORDER}; border-radius:4px;
            }}
        """)
        self.log_box.setFixedHeight(150)
        root.addWidget(self.log_box)
        root.addStretch()

    def _separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color:{C_BORDER};")
        return line

    def current_toggle_states(self) -> dict:
        """Tum toggle'larin guncel (checkbox) durumlari -> worker'a baslangicta uygulanir."""
        return {k: bool(cb.isChecked()) for k, cb in self._toggles.items()}

    def append_log(self, msg: str):
        self.log_box.appendPlainText(msg)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_metrics(self, m: dict):
        total = max(1, m.get("total", 1))
        cur   = m.get("frame", 0)
        acc   = m.get("acc", 0.0)

        self.c_hdg.set_value(f"{m.get('hdg', 0):.1f}")
        self.c_alt.set_value(f"{m.get('alt', 0)}")
        self.c_err.set_value(f"{m.get('err_m', 0)}")
        self.c_spd.set_value(f"{m.get('spd_kmh', 0):.1f}")
        self.c_lat.set_value(f"{m.get('lat', 0):.6f}")
        self.c_lon.set_value(f"{m.get('lon', 0):.6f}")
        self.c_acc.set_value(f"{acc:.1f}")
        self.c_tm.set_value(f"{m.get('max_val2', 0):.3f}")

        err_m = m.get("err_m", 0)
        self.c_err.set_color(C_SUCCESS if err_m < 70 else C_WARN if err_m < 150 else "#f44336")
        self.c_acc.set_color(C_SUCCESS if acc >= 80 else C_WARN if acc >= 50 else "#f44336")

        pct = int(cur / total * 100)
        self.progress.setValue(pct)
        self.progress.setFormat(f"{cur} / {total}  ({pct}%)")


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gorusel Navigasyon — PyQt5")
        self.resize(1400, 900)
        self._apply_dark_theme()

        self._worker: ProcessingWorker | None = None

        # Widget'lar
        self._map         = MapDisplay()
        self._crop_panel  = SmallImagePanel("CROP  (anlık)")
        self._model_panel = SmallImagePanel("MODEL  (sinir ağı)")
        self._panel       = ControlPanel()

        # Sol sütun: Crop üstte, Model altta
        left_col = QSplitter(Qt.Vertical)
        left_col.addWidget(self._crop_panel)
        left_col.addWidget(self._model_panel)
        left_col.setSizes([450, 450])
        left_col.setStyleSheet(f"QSplitter::handle {{ background:{C_BORDER}; height:2px; }}")

        # Ana yatay bölücü: sol sütun | harita | kontrol paneli
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_col)
        splitter.addWidget(self._map)
        splitter.addWidget(self._panel)
        splitter.setSizes([280, 820, 300])
        splitter.setStyleSheet(f"QSplitter::handle {{ background:{C_BORDER}; width:2px; }}")
        self.setCentralWidget(splitter)

        # Durum cubugu
        self._status = QStatusBar()
        self._status.setStyleSheet(f"background:{C_PANEL}; color:{C_MUTED}; font-size:11px;")
        self.setStatusBar(self._status)
        self._status.showMessage("Hazir.")

        # Buton baglantilari
        self._panel.btn_start.clicked.connect(self._on_start)
        self._panel.btn_stop.clicked.connect(self._on_stop)
        self._panel.toggle_changed.connect(self._on_toggle)

    def _apply_dark_theme(self):
        pal = QPalette()
        pal.setColor(QPalette.Window,          QColor(C_BG))
        pal.setColor(QPalette.WindowText,      QColor(C_TEXT))
        pal.setColor(QPalette.Base,            QColor(C_PANEL))
        pal.setColor(QPalette.AlternateBase,   QColor(C_BG))
        pal.setColor(QPalette.Text,            QColor(C_TEXT))
        pal.setColor(QPalette.Button,          QColor(C_PANEL))
        pal.setColor(QPalette.ButtonText,      QColor(C_TEXT))
        pal.setColor(QPalette.Highlight,       QColor(C_ACCENT))
        pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        QApplication.setPalette(pal)

    def _on_start(self):
        if self._worker and self._worker.isRunning():
            return
        self._worker = ProcessingWorker()
        # Baslamadan once yapilan toggle secimlerini (Kalman dahil) worker'a uygula.
        for _k, _v in self._panel.current_toggle_states().items():
            self._worker.set_ui_state(_k, _v)
        self._worker.frame_ready.connect(self._map.update_frame)
        self._worker.crop_ready.connect(self._on_crop_ready)
        self._worker.metrics_update.connect(self._panel.update_metrics)
        self._worker.log_signal.connect(self._panel.append_log)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()
        self._panel.btn_start.setEnabled(False)
        self._panel.btn_stop.setEnabled(True)
        self._status.showMessage("Isleniyor...")

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self._panel.btn_start.setEnabled(True)
        self._panel.btn_stop.setEnabled(False)
        self._status.showMessage("Durduruldu.")

    def _on_toggle(self, key: str, value: bool):
        if self._worker:
            self._worker.set_ui_state(key, value)

    def _on_crop_ready(self, crop_arr, model_arr):
        self._crop_panel.update_frame(crop_arr)
        self._model_panel.update_frame(model_arr)

    def _on_finished(self, result: dict):
        self._panel.btn_start.setEnabled(True)
        self._panel.btn_stop.setEnabled(False)
        self._status.showMessage("Tamamlandi.")
        self._panel.append_log("=== Islem tamamlandi ===")

    def closeEvent(self, event):
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()


# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
