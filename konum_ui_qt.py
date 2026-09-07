"""Mission Control template-matching viewer, using the shared Mission UI shell."""

import importlib.util
import os
import sys
from numbers import Real

import cv2
import numpy as np
from PyQt5.QtCore import QMutex, QMutexLocker, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from mission_ui import APP_STYLE, COLORS, create_footer, create_header, set_status
from mission_ui import MetricCard as SharedMetricCard

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


class ProcessingWorker(QThread):
    frame_ready = pyqtSignal(object)
    crop_ready = pyqtSignal(object, object)
    metrics_update = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._running = True
        self._mutex = QMutex()
        self._ui_state = {
            "trajectory": bool(DRAW_TRAJECTORY),
            "inner_frame": bool(RUN_CFG.get("SHOW_INNER_FRAME", False)),
            "roi_frame": bool(RUN_CFG.get("SHOW_ROI_FRAME", True)),
            "tm_boxes": bool(RUN_CFG.get("SHOW_TM_BOXES", True)),
            "kalman": bool(_tm.USE_KALMAN),
        }

    def set_ui_state(self, key: str, value: bool):
        with QMutexLocker(self._mutex):
            self._ui_state[key] = value

    def get_ui_state(self, key: str, default=False) -> bool:
        with QMutexLocker(self._mutex):
            return self._ui_state.get(key, default)

    def stop(self):
        self._running = False

    def run(self):
        def on_frame(vis_bgr, metadata):
            self.frame_ready.emit(vis_bgr)
            self.metrics_update.emit({k: v for k, v in metadata.items() if k not in ("vis_crop", "vis_model")})
            crop, model = metadata.get("vis_crop"), metadata.get("vis_model")
            if crop is not None and model is not None:
                self.crop_ready.emit(crop, model)

        def ui_state_update():
            with QMutexLocker(self._mutex):
                return dict(self._ui_state)

        result = {}
        try:
            _tm.run_pipeline({
                "on_frame": on_frame,
                "on_log": lambda msg: self.log_signal.emit(str(msg)),
                "stop_check": lambda: not self._running,
                "ui_state_update": ui_state_update,
            })
            result = {"tamamlandi": self._running, "stopped": not self._running}
        except Exception as exc:
            result = {"tamamlandi": False, "error": str(exc)}
            self.log_signal.emit(f"HATA: {exc}")
        self.finished_signal.emit(result)


class MapDisplay(QLabel):
    """Keep the source pixmap so panel resizing never stretches a previous preview."""

    def __init__(self, placeholder="Başlatıldığında operasyon haritası burada görüntülenecek"):
        super().__init__(placeholder)
        self.setObjectName("ImageWell")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(160, 100)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._source = None

    def update_frame(self, arr: np.ndarray):
        if arr is None or arr.size == 0:
            return
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB if arr.ndim == 2 else cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self._source = QPixmap.fromImage(image)
        self._rescale()

    def _rescale(self):
        if self._source is not None:
            self.setPixmap(self._source.scaled(self.contentsRect().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class SmallImagePanel(QFrame):
    def __init__(self, title: str, placeholder: str):
        super().__init__()
        self.setObjectName("EvidenceCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        self._img = MapDisplay(placeholder)
        self._img.setAccessibleName(title)
        layout.addWidget(self._img, 1)

    def update_frame(self, arr):
        self._img.update_frame(arr)


class MetricCard(SharedMetricCard):
    def __init__(self, label: str, unit: str = "", wide: bool = False):
        super().__init__(label)
        self._unit = unit
        self._val = self.value
        if wide:
            self._val.setStyleSheet("font-size:17px;")
        self._val.setWordWrap(True)

    def set_value(self, value):
        self._val.setText("—" if value is None or value == "--" else f"{value}{self._unit}")

    def set_color(self, color: str):
        self._val.setStyleSheet(f"color:{color};")


class ControlPanel(QFrame):
    toggle_changed = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("SidePanel")
        self.setMinimumWidth(290)
        root = QVBoxLayout(self)
        root.setContentsMargins(11, 11, 11, 11)
        root.setSpacing(10)
        title = QLabel("NAVİGASYON")
        title.setObjectName("SectionTitle")
        root.addWidget(title)

        self.controls = QFrame()
        self.controls.setObjectName("ControlGroup")
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(9, 8, 9, 8)
        action_row = QHBoxLayout()
        self.btn_start = QPushButton("Eşleştirmeyi başlat")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.setAccessibleName("Eşleştirmeyi başlat")
        self.btn_stop = QPushButton("Durdur")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        action_row.addWidget(self.btn_start)
        action_row.addWidget(self.btn_stop)
        self._toggles = {}
        view_row = QHBoxLayout()
        view_label = QLabel("GÖRÜNÜM")
        view_label.setObjectName("ControlGroupLabel")
        view_row.addWidget(view_label)
        for key, text, default in (
            ("kalman", "Kalman filtresi", bool(_tm.USE_KALMAN)),
            ("trajectory", "Rota", bool(DRAW_TRAJECTORY)),
            ("roi_frame", "Arama alanı", bool(RUN_CFG.get("SHOW_ROI_FRAME", True))),
            ("tm_boxes", "Eşleşme kutuları", bool(RUN_CFG.get("SHOW_TM_BOXES", True))),
            ("inner_frame", "İç çerçeve", bool(RUN_CFG.get("SHOW_INNER_FRAME", False))),
        ):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setChecked(default)
            button.setProperty("controlRole", "method" if key == "kalman" else "visual")
            button.setAccessibleName(text)
            button.setToolTip("Konum kestirim filtresini açar veya kapatır." if key == "kalman" else "Yalnızca haritadaki çizimlerin görünürlüğünü değiştirir.")
            button.toggled.connect(lambda checked, k=key: self.toggle_changed.emit(k, checked))
            self._toggles[key] = button
            (action_row if key == "kalman" else view_row).addWidget(button)
        action_row.addStretch(1)
        view_row.addStretch(1)
        controls_layout.addLayout(action_row)
        controls_layout.addLayout(view_row)

        self.c_hdg = MetricCard("Başlık", "°")
        self.c_alt = MetricCard("İrtifa", " m")
        self.c_err = MetricCard("Konum hatası", " m")
        self.c_tm = MetricCard("TM skoru")
        self.c_spd = MetricCard("LOC SPD", " km/h")
        self.c_of_spd = MetricCard("OF SPD", " km/h")
        self.c_acc = MetricCard("Başarı oranı", "%")
        self.c_lat = MetricCard("Enlem", wide=True)
        self.c_lon = MetricCard("Boylam", wide=True)
        self.c_spd.setToolTip("Ardışık eşleştirme/Kalman konumlarından hesaplanan hız.")
        self.c_of_spd.setToolTip("Görüntüler arasındaki optik akıştan hesaplanan bağımsız hız.")
        grid = QGridLayout()
        for index, card in enumerate((self.c_hdg, self.c_alt, self.c_err, self.c_tm)):
            grid.addWidget(card, index // 2, index % 2)
        root.addLayout(grid)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Kare bekleniyor")
        self.progress.setAccessibleName("İşlenen kare ilerlemesi")
        root.addWidget(self.progress)

        self.tabs = QTabWidget()
        details = QWidget()
        details_layout = QVBoxLayout(details)
        for card in (self.c_spd, self.c_of_spd, self.c_acc, self.c_lat, self.c_lon):
            details_layout.addWidget(card)
        self.flow_reason = QLabel("Optik akış ölçümü bekleniyor")
        self.flow_reason.setObjectName("Muted")
        self.flow_reason.setWordWrap(True)
        details_layout.addWidget(self.flow_reason)
        details_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(details)
        self.tabs.addTab(scroll, "TELEMETRİ")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setPlaceholderText("İşlem başladığında olaylar burada görüntülenecek.")
        self.log_box.setAccessibleName("Eşleştirme olay günlüğü")
        self.tabs.addTab(self.log_box, "OLAY GÜNLÜĞÜ")
        root.addWidget(self.tabs, 1)

    def current_toggle_states(self) -> dict:
        return {key: button.isChecked() for key, button in self._toggles.items()}

    def append_log(self, message: str):
        self.log_box.appendPlainText(message)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def update_metrics(self, metadata: dict):
        fields = (
            (self.c_hdg, "hdg", 1), (self.c_alt, "alt", 1),
            (self.c_err, "err_m", 1), (self.c_tm, "max_val2", 3),
            (self.c_spd, "spd_kmh", 1), (self.c_acc, "acc", 1),
            (self.c_lat, "lat", 6), (self.c_lon, "lon", 6),
        )
        for card, key, digits in fields:
            value = metadata.get(key)
            card.set_value(f"{value:.{digits}f}" if isinstance(value, Real) and np.isfinite(value) else None)
        valid_flow = metadata.get("of_spd_valid", False)
        flow_speed = metadata.get("of_spd_kmh")
        if valid_flow and isinstance(flow_speed, Real) and np.isfinite(flow_speed):
            self.c_of_spd.set_value(f"{flow_speed:.1f}")
            self.c_of_spd.set_color(COLORS["success"])
        else:
            self.c_of_spd.set_value(None)
            self.c_of_spd.set_color(COLORS["muted"])
        reasons = {
            "ok": "ölçüm geçerli", "warmup": "ikinci kare bekleniyor",
            "shape_changed": "görüntü boyutu değişti", "timestamp_missing": "zaman bilgisi yok",
            "invalid_dt": "kareler arası süre geçersiz", "invalid_gsd": "yer örnekleme aralığı geçersiz",
            "few_features": "yeterli özellik noktası yok", "flow_failed": "nokta takibi başarısız",
            "few_tracks": "takip edilen nokta sayısı yetersiz", "model_failed": "hareket kestirilemedi",
            "low_inlier_ratio": "hareket tutarlılığı düşük", "non_finite_speed": "hız hesaplanamadı",
        }
        self.flow_reason.setText("Optik akış: " + reasons.get(metadata.get("of_reason"), "ölçüm bekleniyor"))
        for card, key in ((self.c_err, "err_m"), (self.c_acc, "acc")):
            value = metadata.get(key)
            tone = "muted"
            if isinstance(value, Real) and np.isfinite(value):
                if key == "err_m":
                    tone = "success" if value < 70 else "warning" if value < 150 else "danger"
                else:
                    tone = "success" if value >= 80 else "warning" if value >= 50 else "danger"
            card.set_color(COLORS[tone])
        total = max(1, metadata.get("total", 1))
        current = metadata.get("frame", 0)
        self.progress.setValue(max(0, min(100, int(current / total * 100))))
        self.progress.setFormat(f"{current} / {total} kare  ·  %p%")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mission Control | Template Matching")
        self.resize(1480, 920)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(APP_STYLE)
        self._worker = None
        self._close_requested = False
        root = QWidget()
        root.setObjectName("Root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(10)
        self.setCentralWidget(root)
        header, header_layout, self.status_pill = create_header(
            "TEMPLATE MATCHING", "Görüntü eşleştirme ile konumlama  /  Görsel navigasyon"
        )
        self._panel = ControlPanel()
        header_layout.addWidget(self._panel.controls)
        layout.addWidget(header)
        evidence = QFrame()
        evidence.setObjectName("SidePanel")
        evidence_layout = QVBoxLayout(evidence)
        evidence_layout.setContentsMargins(11, 11, 11, 11)
        title = QLabel("GÖRSEL KANIT")
        title.setObjectName("SectionTitle")
        evidence_layout.addWidget(title)
        self._crop_panel = SmallImagePanel("GÖZLEM", "Canlı görüntü bekleniyor")
        self._model_panel = SmallImagePanel("MODEL", "Sinir ağı çıktısı bekleniyor")
        self.evidence_splitter = QSplitter(Qt.Vertical)
        self.evidence_splitter.setChildrenCollapsible(False)
        self.evidence_splitter.addWidget(self._crop_panel)
        self.evidence_splitter.addWidget(self._model_panel)
        evidence_layout.addWidget(self.evidence_splitter, 1)
        map_panel = QFrame()
        map_panel.setObjectName("MapPanel")
        map_layout = QVBoxLayout(map_panel)
        map_layout.setContentsMargins(10, 10, 10, 10)
        title = QLabel("OPERASYON HARİTASI")
        title.setObjectName("SectionTitle")
        map_layout.addWidget(title)
        self._map = MapDisplay()
        self._map.setAccessibleName("Operasyon haritası")
        map_layout.addWidget(self._map, 1)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setAccessibleName("Ana panel genişlik ayırıcısı")
        for widget in (evidence, map_panel, self._panel):
            self.main_splitter.addWidget(widget)
        self.main_splitter.setSizes([280, 820, 320])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        layout.addWidget(self.main_splitter, 1)
        footer, self.performance_label = create_footer(
            "Görünüm kontrolleri üstte    ·    Ayrıntılar ve günlük sağda    |    Panel sınırlarını sürükleyin"
        )
        layout.addWidget(footer)
        self._panel.btn_start.clicked.connect(self._on_start)
        self._panel.btn_stop.clicked.connect(self._on_stop)
        self._panel.toggle_changed.connect(self._on_toggle)

    def _on_start(self):
        if self._worker and self._worker.isRunning():
            return
        self._worker = ProcessingWorker()
        for key, value in self._panel.current_toggle_states().items():
            self._worker.set_ui_state(key, value)
        self._worker.frame_ready.connect(self._map.update_frame)
        self._worker.crop_ready.connect(self._on_crop_ready)
        self._worker.log_signal.connect(self._panel.append_log)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.finished.connect(self._on_worker_stopped)
        self._panel.btn_start.setEnabled(False)
        self._panel.btn_stop.setEnabled(True)
        self._panel.update_metrics({})
        self._panel.progress.setRange(0, 0)
        self._worker.metrics_update.connect(self._on_progress)
        set_status(self.status_pill, "İŞLENİYOR", "accent")
        self._worker.start()

    def _on_progress(self, metadata):
        self._panel.progress.setRange(0, 100)
        self._panel.update_metrics(metadata)
        elapsed = metadata.get("calisma_s")
        if isinstance(elapsed, Real) and np.isfinite(elapsed):
            self.performance_label.setText(f"{metadata.get('frame', 0)} kare  ·  {elapsed:.1f} s")

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._panel.btn_stop.setEnabled(False)
            set_status(self.status_pill, "DURDURULUYOR", "warning")

    def _on_toggle(self, key: str, value: bool):
        if self._worker:
            self._worker.set_ui_state(key, value)

    def _on_crop_ready(self, crop_arr, model_arr):
        self._crop_panel.update_frame(crop_arr)
        self._model_panel.update_frame(model_arr)

    def _on_finished(self, result: dict):
        self._panel.progress.setRange(0, 100)
        if result.get("error"):
            set_status(self.status_pill, "HATA", "danger")
            self._panel.tabs.setCurrentIndex(1)
        elif result.get("stopped"):
            set_status(self.status_pill, "DURDURULDU")
            self._panel.append_log("İşlem kullanıcı tarafından durduruldu.")
        else:
            set_status(self.status_pill, "TAMAMLANDI", "success")
            self._panel.append_log("İşlem tamamlandı.")

    def _on_worker_stopped(self):
        self._panel.btn_start.setEnabled(True)
        self._panel.btn_stop.setEnabled(False)
        if self._close_requested:
            self.close()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._close_requested = True
            self._on_stop()
            event.ignore()
            return
        event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
