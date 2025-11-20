import cv2
import os
import numpy as np
import csv
import json
import atexit
from datetime import datetime

# Import các module logic
from detector import VehicleDetector
from sort import Sort
from counter import Counter

# Import ClickableLabel để dùng chung
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QPoint, QRect


# ============================================================
# Label hỗ trợ chọn ROI bằng chuột
# (Đã di chuyển từ gui.py sang đây)
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

    def clear_roi(self):
        self._has_roi = False
        self._roi_rect_frame = None
        self.update()

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
                    self.setToolTip(f"ROI: {roi_frame}")
                    self._roi_callback(roi_frame)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        # Vẽ ROI đang vẽ (màu xanh lá)
        if self._drawing:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
            painter.drawRect(self._current_rect)

        # Vẽ ROI đã chốt (màu cam)
        if self._has_roi and self._roi_rect_frame:
            disp_rect = self.map_frame_rect_to_display(self._roi_rect_frame)
            painter.setPen(QPen(QColor(255, 165, 0), 2, Qt.PenStyle.DashLine))
            painter.drawRect(disp_rect)

    def map_rect_to_frame(self, disp_rect: QRect):
        pix = self.pixmap()
        if not self._frame_size or not pix or pix.isNull() or pix.width() == 0 or pix.height() == 0:
            return None
        lbl_w, lbl_h = self.width(), self.height()
        disp_w, disp_h = pix.width(), pix.height()
        scale = min(lbl_w / disp_w, lbl_h / disp_h)
        if scale == 0: return None
        new_w, new_h = disp_w * scale, disp_h * scale
        off_x, off_y = (lbl_w - new_w) / 2, (lbl_h - new_h) / 2
        x1, y1 = max(0, disp_rect.left() - off_x), max(0, disp_rect.top() - off_y)
        x2, y2 = disp_rect.right() - off_x, disp_rect.bottom() - off_y
        frame_w, frame_h = self._frame_size
        fx1 = int(x1 * frame_w / new_w)
        fy1 = int(y1 * frame_h / new_w)
        fx2 = int(x2 * frame_w / new_w)
        fy2 = int(y2 * frame_h / new_w)
        return (max(0, fx1), max(0, fy1), min(frame_w, fx2), min(frame_h, fy2))

    def map_frame_rect_to_display(self, rect):
        pix = self.pixmap()
        if not rect or not self._frame_size or not pix or pix.isNull() or pix.width() == 0 or pix.height() == 0:
            return QRect()
        x1, y1, x2, y2 = rect
        frame_w, frame_h = self._frame_size
        lbl_w, lbl_h = self.width(), self.height()
        disp_w, disp_h = pix.width(), pix.height()
        scale = min(lbl_w / disp_w, lbl_h / disp_h)
        if scale == 0: return QRect()
        new_w, new_h = disp_w * scale, disp_h * scale
        off_x, off_y = (lbl_w - new_w) / 2, (lbl_h - new_h) / 2
        dx1 = int(x1 * new_w / frame_w + off_x)
        dy1 = int(y1 * new_h / frame_h + off_y)
        dx2 = int(x2 * new_w / frame_w + off_x)
        dy2 = int(y2 * new_h / frame_h + off_y)
        return QRect(dx1, dy1, dx2 - dx1, dy2 - dy1)


# ============================================================
# Class Engine xử lý video
# ============================================================
class VideoEngine:
    def __init__(self, model_path="yolov8n.pt"):
        # Khởi tạo các module
        self.detector = VehicleDetector(model_path=model_path)
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        self.counter_down = None
        self.counter_up = None

        # Biến trạng thái
        self.cap = None
        self.frame_idx = 0
        self.video_path = None
        self.roi = None

        # Biến lưu trữ
        self.prev_centroids = {}
        self.id_classes = {}  # Bộ nhớ lưu class của ID

        # Biến xử lý file output
        self.output_dir = "outputs"
        self.csv_path = None
        self.summary_path = None
        self._csv_file = None
        self._csv_writer = None
        os.makedirs(self.output_dir, exist_ok=True)
        atexit.register(self.stop)  # Đảm bảo file được đóng khi thoát

    def is_running(self):
        return self.cap is not None and self.cap.isOpened()

    def set_roi(self, roi_rect):
        self.roi = roi_rect

    def start(self, video_path):
        """
        Bắt đầu xử lý video.
        Trả về (width, height) nếu thành công.
        """
        self.stop()  # Dừng video cũ (nếu có)

        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.cap = None
            raise FileNotFoundError(f"Không thể mở video: {video_path}")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Cấu hình line đếm
        self.line_down_y = int(height * 0.5)
        self.line_up_y = int(height * 0.6)

        # Khởi tạo/Reset các module
        self.tracker = Sort(max_age=90, min_hits=2, iou_threshold=0.1)
        self.counter_down = Counter(line_position_y=self.line_down_y, direction="down")
        self.counter_up = Counter(line_position_y=self.line_up_y, direction="up")

        # Reset trạng thái
        self.frame_idx = 0
        self.prev_centroids = {}
        self.id_classes = {}

        # Mở file CSV
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        self.csv_path = f"{self.output_dir}/{base_name}_counts.csv"
        self.summary_path = f"{self.output_dir}/{base_name}_summary.json"

        self._open_csv()

        return (width, height)

    def stop(self):
        """
        Dừng xử lý và lưu file summary.
        Trả về đường dẫn file summary.
        """
        if self.cap:
            self.cap.release()
            self.cap = None

        self._close_csv()

        summary_path_to_return = None

        # Chỉ lưu summary nếu 2 bộ đếm đã được khởi tạo
        if self.counter_down and self.counter_up:
            summary_down = self.counter_down.get_summary()
            summary_up = self.counter_up.get_summary()

            combined_counts = {}
            all_keys = set(summary_down["counts"].keys()) | set(summary_up["counts"].keys())
            for k in all_keys:
                combined_counts[k] = summary_down["counts"].get(k, 0) + summary_up["counts"].get(k, 0)

            summary = {
                "source_video": self.video_path,
                "total_all": sum(combined_counts.values()),
                "counts_by_class_total": combined_counts,
                "details": {
                    "total_down": summary_down["total"],
                    "counts_down": summary_down["counts"],
                    "total_up": summary_up["total"],
                    "counts_up": summary_up["counts"]
                }
            }

            try:
                with open(self.summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                summary_path_to_return = self.summary_path
            except Exception as e:
                print(f"Lỗi khi lưu JSON: {e}")

        self.video_path = None
        return summary_path_to_return

    def process_next_frame(self):
        """
        Xử lý frame tiếp theo.
        Trả về: (ret, frame_vẽ, stats_dict)
        """
        if not self.is_running():
            return False, None, {}

        ret, frame = self.cap.read()
        if not ret:
            return False, None, {}

        frame_to_show = frame.copy()
        self.frame_idx += 1

        # 1. Detect
        detections = self.detector.detect(frame, conf=0.4)

        # 2. Chuẩn bị data cho SORT (Numpy)
        dets_to_sort = []
        current_frame_classes = {}  # Map tạm

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            score = d.get("conf", 0.5)
            cls_name = d["cls_name"]

            # Lọc bằng ROI (nếu có)
            if self.roi:
                rx1, ry1, rx2, ry2 = self.roi
                # Tính trung tâm của box
                box_cx = (x1 + x2) / 2
                box_cy = (y1 + y2) / 2
                # Chỉ xử lý nếu trung tâm nằm trong ROI
                if not (rx1 < box_cx < rx2 and ry1 < box_cy < ry2):
                    continue

            dets_to_sort.append([x1, y1, x2, y2, score])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            current_frame_classes[(cx, cy)] = cls_name  # Lưu class theo tọa độ

        dets_to_sort = np.array(dets_to_sort)
        if len(dets_to_sort) == 0:
            dets_to_sort = np.empty((0, 5))

        # 3. Update Tracker (SORT)
        tracked_dets = self.tracker.update(dets_to_sort)

        # 4. Loop qua các xe đã track
        for track in tracked_dets:
            x1, y1, x2, y2, oid = track
            oid = int(oid)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cX, cY = int((x1 + x2) / 2), int((y1 + y2) / 2)
            curr_centroid = (cX, cY)

            # 5. Tìm lại Class Name (vì SORT không lưu)
            if oid not in self.id_classes:
                min_dist = 100
                assigned_cls = "unknown"
                for (dcx, dcy), name in current_frame_classes.items():
                    dist = ((dcx - cX) ** 2 + (dcy - cY) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        assigned_cls = name
                self.id_classes[oid] = assigned_cls

            cls_name = self.id_classes.get(oid, "car")

            # 6. Đếm
            prev = self.prev_centroids.get(oid, curr_centroid)
            timestamp = datetime.now().isoformat()

            c_down = self.counter_down.check_and_count(oid, prev, curr_centroid, cls_name, self.frame_idx, timestamp)
            c_up = self.counter_up.check_and_count(oid, prev, curr_centroid, cls_name, self.frame_idx, timestamp)
            counted = c_down or c_up

            self.prev_centroids[oid] = curr_centroid

            # 7. Ghi CSV
            if c_down:
                self._write_csv_row([self.frame_idx, oid, cls_name, "down", timestamp])
            elif c_up:
                self._write_csv_row([self.frame_idx, oid, cls_name, "up", timestamp])

            # 8. Vẽ
            color = (0, 255, 0) if not counted else (0, 0, 255)
            cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), color, 2)
            label = f"ID {oid} {cls_name}"
            cv2.putText(frame_to_show, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.circle(frame_to_show, (cX, cY), 4, color, -1)

        # 9. Lấy số liệu thống kê hiện tại
        stats = self.get_stats()

        # 10. Vẽ vạch đếm + ROI
        cv2.line(frame_to_show, (0, self.line_down_y), (frame.shape[1], self.line_down_y), (0, 255, 255), 2)  # Vàng
        cv2.line(frame_to_show, (0, self.line_up_y), (frame.shape[1], self.line_up_y), (255, 0, 255), 2)  # Tím
        if self.roi:
            rx1, ry1, rx2, ry2 = self.roi
            cv2.rectangle(frame_to_show, (rx1, ry1), (rx2, ry2), (255, 165, 0), 2)
            cv2.putText(frame_to_show, "ROI", (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # 11. Chuyển đổi màu BGR sang RGB để PyQt hiển thị
        frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)

        return True, frame_rgb, stats

    def get_stats(self):
        """Lấy số liệu tổng hợp từ cả 2 bộ đếm."""
        if not self.counter_down or not self.counter_up:
            return {}

        counts_down = self.counter_down.counts
        counts_up = self.counter_up.counts

        combined_counts = {}
        all_keys = set(counts_down.keys()) | set(counts_up.keys())
        for k in all_keys:
            combined_counts[k] = counts_down.get(k, 0) + counts_up.get(k, 0)

        combined_counts["total"] = sum(combined_counts.values())
        return combined_counts

    # --- CSV Helper Functions ---
    def _open_csv(self):
        try:
            self._close_csv()  # Đóng file cũ nếu có
            first_write = not os.path.exists(self.csv_path)
            self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            if first_write:
                self._csv_writer.writerow(["frame", "object_id", "class", "direction", "timestamp"])
            self._csv_file.flush()
        except Exception as e:
            print(f"Lỗi khi mở CSV: {e}")

    def _write_csv_row(self, row):
        if self._csv_writer:
            try:
                self._csv_writer.writerow(row)
                self._csv_file.flush()
            except Exception as e:
                print(f"Lỗi khi ghi CSV: {e}")

    def _close_csv(self):
        if self._csv_file and not self._csv_file.closed:
            try:
                self._csv_file.close()
            except Exception as e:
                print(f"Lỗi khi đóng CSV: {e}")
        self._csv_file = None
        self._csv_writer = None