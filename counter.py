# counter.py
from datetime import datetime
import csv
import os


class Counter:
    def __init__(self, line_position_y, direction="down"):
        """
        line_position_y: pixel y của counting line
        direction: "down" (từ trên xuống) hoặc "up" (từ dưới lên)
        """
        self.line_y = int(line_position_y)
        self.direction = direction
        self.counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        self.counted_ids = set()
        self.history = []

    def reset(self):
        self.counts = {k: 0 for k in self.counts.keys()}
        self.counted_ids = set()
        self.history = []

    def set_line(self, y):
        self.line_y = int(y)

    def check_and_count(self, object_id, prev_centroid, curr_centroid, cls_name, frame_idx, timestamp=None):
        """
        Trả về True nếu object vừa được đếm.
        """
        # 1. Nếu đã đếm rồi thì bỏ qua ngay
        if object_id in self.counted_ids:
            return False

        # 2. Đảm bảo có key class trong dict
        if cls_name not in self.counts:
            self.counts[cls_name] = 0

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prev_y = prev_centroid[1]
        curr_y = curr_centroid[1]

        should_count = False

        # --- LOGIC ĐẾM ---
        if self.direction == "down":
            # Đi xuống: Trước đó ở trên (nhỏ hơn) -> Giờ ở dưới (lớn hơn hoặc bằng)
            if prev_y < self.line_y and curr_y >= self.line_y:
                should_count = True

        elif self.direction == "up":
            # Đi lên: Trước đó ở dưới (lớn hơn) -> Giờ ở trên (nhỏ hơn hoặc bằng)
            if prev_y > self.line_y and curr_y <= self.line_y:
                should_count = True

        # --- XỬ LÝ KHI ĐẾM THÀNH CÔNG ---
        if should_count:
            self.counts[cls_name] += 1
            self.counted_ids.add(object_id)
            self.history.append((frame_idx, object_id, cls_name, timestamp))

            # DEBUG: In ra để biết nó có hoạt động không
            print(f"✅ COUNTED {self.direction.upper()}: ID {object_id} ({cls_name})")
            return True

        return False

    def get_summary(self):
        total = sum(self.counts.values())
        return {"counts": dict(self.counts), "total": total}