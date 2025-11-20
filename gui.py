import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QMessageBox, QGroupBox, QFormLayout
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect

# Import ClickableLabel và Engine xử lý từ file logic
from video_engine import ClickableVideoLabel, VideoEngine


# ============================================================
# GUI chính
# ============================================================
class VehicleCounterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 Vehicle Counter - YOLOv8 + SORT (Refactored)")
        self.setGeometry(100, 60, 1280, 720)

        # Khởi tạo Engine xử lý
        # Engine sẽ lo toàn bộ logic nặng
        self.engine = VideoEngine()

        # 1. CỘT TRÁI (VIDEO)
        self.video_label = ClickableVideoLabel(self)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Vui lòng chọn video và nhấn Bắt đầu")
        # Kết nối callback ROI từ Label tới Engine
        self.video_label.set_roi_callback(self.set_roi)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)

        # 2. CỘT PHẢI (ĐIỀU KHIỂN & KẾT QUẢ)
        # Layout cột phải
        right_layout = QVBoxLayout()
        control_group = QGroupBox("Bảng điều khiển")
        control_layout = QVBoxLayout()
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
        control_layout.addWidget(self.btn_open)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_clear_roi)
        control_layout.addWidget(self.btn_exit)
        control_group.setLayout(control_layout)

        # Layout bảng kết quả
        results_group = QGroupBox("Kết quả đếm (Tổng 2 chiều)")
        results_layout = QFormLayout()
        self.total_label = QLabel("0")
        self.car_label = QLabel("0")
        self.truck_label = QLabel("0")
        self.bus_label = QLabel("0")
        self.motorcycle_label = QLabel("0")
        label_style = "font-size: 16px; font-weight: bold; color: #333;"
        self.total_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #d9534f;")
        self.car_label.setStyleSheet(label_style)
        self.truck_label.setStyleSheet(label_style)
        self.bus_label.setStyleSheet(label_style)
        self.motorcycle_label.setStyleSheet(label_style)
        results_layout.addRow("TỔNG CỘNG:", self.total_label)
        results_layout.addRow("Car:", self.car_label)
        results_layout.addRow("Truck:", self.truck_label)
        results_layout.addRow("Bus:", self.bus_label)
        results_layout.addRow("Motorcycle:", self.motorcycle_label)
        results_group.setLayout(results_layout)

        right_layout.addWidget(control_group)
        right_layout.addStretch(1)
        right_layout.addWidget(results_group)
        right_layout.addStretch(1)

        # 3. LAYOUT CHÍNH (QHBoxLayout)
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 3)  # Cột video chiếm 3 phần
        main_layout.addLayout(right_layout, 1)  # Cột control chiếm 1 phần
        self.setLayout(main_layout)

        # Biến trạng thái của GUI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.video_path = None
        self.paused = False

    def open_file(self):
        """Mở dialog chọn file video."""
        file, _ = QFileDialog.getOpenFileName(self, "Chọn video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file:
            self.video_path = file
            self.video_label.setText(f"Đã chọn: {os.path.basename(file)}")

    def start_video(self):
        """Bắt đầu xử lý video (giao việc cho Engine)."""
        if not self.video_path:
            QMessageBox.warning(self, "Thông báo", "Vui lòng chọn video trước.")
            return

        # Giao việc cho Engine
        try:
            # Engine sẽ mở video và trả về kích thước
            width, height = self.engine.start(self.video_path)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể bắt đầu xử lý: {e}")
            return

        # Cài đặt kích thước cho label (để tính toán ROI)
        self.video_label.set_frame_size(width, height)

        # Reset các label đếm về 0
        self.total_label.setText("0")
        self.car_label.setText("0")
        self.truck_label.setText("0")
        self.bus_label.setText("0")
        self.motorcycle_label.setText("0")

        # Khởi động timer để cập nhật frame
        self.paused = False
        self.timer.start(30)  # 30ms ~ 33 FPS
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸️ Pause")

    def toggle_pause(self):
        """Tạm dừng hoặc tiếp tục timer."""
        if not self.engine.is_running():
            return

        self.paused = not self.paused  # Đảo trạng thái

        if self.paused:
            self.timer.stop()
            self.btn_pause.setText("▶️ Resume")
        else:
            self.timer.start(30)
            self.btn_pause.setText("⏸️ Pause")

    def clear_roi(self):
        """Xóa vùng ROI đã chọn."""
        self.engine.set_roi(None)  # Báo cho Engine
        self.video_label.clear_roi()  # Yêu cầu Label vẽ lại
        QMessageBox.information(self, "ROI", "Đã xóa vùng ROI.")

    def set_roi(self, roi_rect):
        """Callback khi người dùng vẽ ROI xong."""
        self.engine.set_roi(roi_rect)  # Gửi vùng ROI cho Engine
        QMessageBox.information(self, "ROI", f"Đã chọn ROI: {roi_rect}")

    def update_frame(self):
        """Hàm chính, được gọi liên tục bởi QTimer."""
        # 1. Yêu cầu Engine xử lý frame tiếp theo
        # Engine trả về: (True/False, ảnh đã vẽ, dict số liệu)
        ret, frame, stats = self.engine.process_next_frame()

        # 2. Nếu hết video (ret=False)
        if not ret:
            self.end_video()
            return

        # 3. Cập nhật bảng kết quả
        self.total_label.setText(str(stats.get("total", 0)))
        self.car_label.setText(str(stats.get("car", 0)))
        self.truck_label.setText(str(stats.get("truck", 0)))
        self.bus_label.setText(str(stats.get("bus", 0)))
        self.motorcycle_label.setText(str(stats.get("motorcycle", 0)))

        # 4. Hiển thị frame lên GUI
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def end_video(self):
        """Dừng timer và yêu cầu Engine lưu kết quả."""
        self.timer.stop()

        # Yêu cầu Engine dừng và lưu kết quả
        summary_path = self.engine.stop()

        if summary_path:
            QMessageBox.information(self, "Hoàn thành",
                                    f"✅ Video đã xử lý xong.\nKết quả lưu tại: {summary_path}")
        else:
            QMessageBox.information(self, "Hoàn thành", "Đã xử lý xong.")

        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.video_label.setText("Hoàn thành! Vui lòng chọn video mới.")

    def closeEvent(self, event):
        """Đảm bảo Engine dừng khi đóng cửa sổ."""
        self.engine.stop()  # Đảm bảo engine đã dừng
        event.accept()


# ============================================================
# Main (Điểm bắt đầu chạy)
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VehicleCounterGUI()
    gui.show()
    sys.exit(app.exec())