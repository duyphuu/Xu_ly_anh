import sys
import os
import cv2
import csv
import json
import atexit
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QMessageBox, QSizePolicy, QFrame
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import logic
from detector import VehicleDetector
from tracker import CentroidTracker
from counter import Counter


# ============================================================
# Label hỗ trợ chọn ROI bằng chuột
# ============================================================
class ClickableVideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._drawing = False
        self._start_pos = QPoint()
        self._current_rect = QRect()
        self._frame_size = None
        self._roi_callback = None
        self._has_roi = False
        self._roi_rect_frame = None

    def set_frame_size(self, frame_w, frame_h):
        self._frame_size = (frame_w, frame_h)

    def set_roi_callback(self, cb):
        self._roi_callback = cb

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap():
            self._drawing = True
            self._start_pos = event.pos()
            self._current_rect = QRect(self._start_pos, self._start_pos)
            self.update()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._current_rect = QRect(self._start_pos, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            self._current_rect = QRect(self._start_pos, event.pos()).normalized()
            roi_frame = self.map_rect_to_frame(self._current_rect)
            if roi_frame:
                self._has_roi = True
                self._roi_rect_frame = roi_frame
                if self._roi_callback:
                    self._roi_callback(roi_frame)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self._drawing:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
            painter.drawRect(self._current_rect)
        if self._has_roi and self._roi_rect_frame:
            disp_rect = self.map_frame_rect_to_display(self._roi_rect_frame)
            painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.PenStyle.DashLine))
            painter.drawRect(disp_rect)

    def map_rect_to_frame(self, disp_rect: QRect):
        if not self._frame_size or not self.pixmap():
            return None
        lbl_w, lbl_h = self.width(), self.height()
        pix = self.pixmap()
        disp_w, disp_h = pix.width(), pix.height()
        scale = min(lbl_w / disp_w, lbl_h / disp_h)
        if scale == 0:
            return None
        new_w, new_h = disp_w * scale, disp_h * scale
        off_x, off_y = (lbl_w - new_w) / 2, (lbl_h - new_h) / 2
        x1, y1 = max(0, disp_rect.left() - off_x), max(0, disp_rect.top() - off_y)
        x2, y2 = disp_rect.right() - off_x, disp_rect.bottom() - off_y
        frame_w, frame_h = self._frame_size
        fx1 = int(x1 * frame_w / new_w)
        fy1 = int(y1 * frame_h / new_h)
        fx2 = int(x2 * frame_w / new_w)
        fy2 = int(y2 * frame_h / new_h)
        return (max(0, fx1), max(0, fy1), min(frame_w, fx2), min(frame_h, fy2))

    def map_frame_rect_to_display(self, rect):
        if not rect or not self._frame_size or not self.pixmap():
            return QRect()
        x1, y1, x2, y2 = rect
        frame_w, frame_h = self._frame_size
        lbl_w, lbl_h = self.width(), self.height()
        pix = self.pixmap()
        disp_w, disp_h = pix.width(), pix.height()
        scale = min(lbl_w / disp_w, lbl_h / disp_h)
        new_w, new_h = disp_w * scale, disp_h * scale
        off_x, off_y = (lbl_w - new_w) / 2, (lbl_h - new_h) / 2
        dx1 = int(x1 * new_w / frame_w + off_x)
        dy1 = int(y1 * new_h / frame_h + off_y)
        dx2 = int(x2 * new_w / frame_w + off_x)
        dy2 = int(y2 * new_h / frame_h + off_y)
        return QRect(dx1, dy1, dx2 - dx1, dy2 - dy1)


# ============================================================
# GUI chính
# ============================================================
class VehicleCounterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 Vehicle Counter - YOLOv8 + PyQt6")
        self.setGeometry(100, 60, 1250, 720)

        # Video label
        self.video_label = ClickableVideoLabel(self)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label = QLabel("Chưa có video đang chạy")
        self.info_label.setStyleSheet("font-size:16px; padding:6px;")

        # Buttons
        self.btn_open = QPushButton("📂 Chọn video")
        self.btn_open.clicked.connect(self.open_file)
        self.btn_start = QPushButton("▶️ Bắt đầu")
        self.btn_start.clicked.connect(self.start_video)
        self.btn_pause = QPushButton("⏸️ Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_clear_roi = QPushButton("🧽 Xóa ROI")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.btn_exit = QPushButton("❌ Thoát")
        self.btn_exit.clicked.connect(self.close)

        btn_layout = QHBoxLayout()
        for b in [self.btn_open, self.btn_start, self.btn_pause, self.btn_clear_roi, self.btn_exit]:
            btn_layout.addWidget(b)

        # Chart
        self.canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.ax = self.canvas.figure.add_subplot(111)

        chart_frame = QFrame()
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.addWidget(self.canvas)

        # Layout
        main_layout = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self.video_label)
        left.addWidget(self.info_label)
        left.addLayout(btn_layout)
        main_layout.addLayout(left, 3)
        main_layout.addWidget(chart_frame, 1)
        self.setLayout(main_layout)

        # Logic vars
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.detector = None
        self.tracker = None
        self.counter = None
        self.paused = False
        self.video_path = None
        self.frame_idx = 0
        self.line_y = None
        self.roi = None
        self.prev_centroids = {}
        self.video_label.set_roi_callback(self.set_roi)

        # CSV handling
        os.makedirs("outputs", exist_ok=True)
        self.csv_path = None
        self._csv_file = None
        self._csv_writer = None
        atexit.register(self._close_csv)

    # CSV helpers
    def _open_csv(self, video_name):
        base = os.path.splitext(os.path.basename(video_name))[0]
        self.csv_path = f"outputs/{base}_counts.csv"
        first_write = not os.path.exists(self.csv_path)
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if first_write:
            self._csv_writer.writerow(["frame", "object_id", "class"])
        self._csv_file.flush()

    def _close_csv(self):
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

    # GUI actions
    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file:
            self.video_path = file
            self.info_label.setText(f"Đã chọn video: {file}")

    def start_video(self):
        if not self.video_path:
            QMessageBox.warning(self, "Thông báo", "Vui lòng chọn video trước.")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Lỗi", "Không thể mở video.")
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_label.set_frame_size(width, height)
        self.line_y = height // 2

        # modules
        self.detector = VehicleDetector()
        self.tracker = CentroidTracker(max_disappeared=40, max_distance=60)
        self.counter = Counter(line_position_y=self.line_y)
        self.counter.reset()

        self.prev_centroids = {}
        self.frame_idx = 0
        self.paused = False
        self._open_csv(self.video_path)

        self.timer.start(30)
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)

    def toggle_pause(self):
        if not self.cap:
            return
        if self.paused:
            self.timer.start(30)
            self.paused = False
            self.btn_pause.setText("⏸️ Pause")
        else:
            self.timer.stop()
            self.paused = True
            self.btn_pause.setText("▶️ Resume")

    def clear_roi(self):
        self.roi = None
        self.video_label._has_roi = False
        self.video_label.update()
        QMessageBox.information(self, "ROI", "Đã xóa vùng ROI.")

    def set_roi(self, roi_rect):
        self.roi = roi_rect
        QMessageBox.information(self, "ROI", f"Đã chọn ROI: {roi_rect}")

    # Main loop
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.end_video()
            return

        self.frame_idx += 1
        detections = self.detector.detect(frame, conf=0.35)
        rects, cls_map = [], {}
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            if self.roi:
                rx1, ry1, rx2, ry2 = self.roi
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue
            rects.append([x1, y1, x2, y2])
            cls_map[tuple([x1, y1, x2, y2])] = d["cls_name"]

        tracked = self.tracker.update(rects)
        for oid, bbox in tracked.items():
            x1, y1, x2, y2 = bbox
            cX, cY = (x1 + x2)//2, (y1 + y2)//2
            prev = self.prev_centroids.get(oid, (cX, cY))
            cls_name = cls_map.get(tuple(bbox), "car")

            counted = self.counter.check_and_count(oid, prev, (cX, cY), cls_name, frame_idx=self.frame_idx)
            self.prev_centroids[oid] = (cX, cY)

            if counted:
                self._csv_writer.writerow([self.frame_idx, oid, cls_name])
                self._csv_file.flush()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {oid} {cls_name}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.circle(frame, (cX, cY), 3, (0,0,255), -1)

        cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (255,0,0), 2)

        total = sum(self.counter.counts.values())
        self.info_label.setText(f"Tổng: {total} | " + " | ".join(f"{k}:{v}" for k,v in self.counter.counts.items()))
        self.update_chart(self.counter.counts)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = frame_rgb.shape
        img = QImage(frame_rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def update_chart(self, counts):
        self.ax.clear()
        self.ax.bar(counts.keys(), counts.values(), color="skyblue")
        self.ax.set_title("Số lượng theo loại xe")
        self.canvas.draw()

    def end_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self._close_csv()

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        summary_path = f"outputs/{base}_summary.json"
        summary = self.counter.get_summary()
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        QMessageBox.information(self, "Hoàn thành", f"✅ Video '{base}' đã xử lý xong.\nKết quả lưu trong thư mục 'outputs/'.")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.info_label.setText("Sẵn sàng chọn video mới.")

    def closeEvent(self, event):
        self._close_csv()
        if self.cap:
            self.cap.release()
        event.accept()


# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VehicleCounterGUI()
    gui.show()
    sys.exit(app.exec())
