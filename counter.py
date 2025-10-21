# counter.py
# Logic đếm: khi centroid của object vượt qua counting line (dòng ngang), tăng count
from datetime import datetime
import csv
import os

class Counter:
    def __init__(self, line_position_y, direction="down"):
        """
        line_position_y: pixel y của counting line
        direction: "down" (từ trên xuống) hoặc "up"
        """
        self.line_y = line_position_y
        self.direction = direction
        self.counts = {"car":0, "motorcycle":0, "bus":0, "truck":0}
        self.counted_ids = set()
        self.history = []  # lưu sự kiện (frame_idx, id, cls_name, timestamp)

    def reset(self):
        """Reset bộ đếm để bắt đầu video mới."""
        self.counts = {k: 0 for k in self.counts.keys()}
        self.counted_ids = set()
        self.history = []

    def set_line(self, y):
        """Cập nhật vị trí line đếm (pixel y)."""
        self.line_y = int(y)

    def check_and_count(self, object_id, prev_centroid, curr_centroid, cls_name, frame_idx, timestamp=None):
        """
        Trả về True nếu object vừa được đếm (mới tăng).
        prev_centroid, curr_centroid: (x,y)
        """
        if object_id in self.counted_ids:
            return False

        # ensure class key exists
        if cls_name not in self.counts:
            self.counts.setdefault(cls_name, 0)

        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        # direction down: y tăng when moving down
        if self.direction == "down":
            if prev_centroid[1] < self.line_y and curr_centroid[1] >= self.line_y:
                self.counts[cls_name] = self.counts.get(cls_name, 0) + 1
                self.counted_ids.add(object_id)
                self.history.append((frame_idx, object_id, cls_name, timestamp))
                return True
        else:
            if prev_centroid[1] > self.line_y and curr_centroid[1] <= self.line_y:
                self.counts[cls_name] = self.counts.get(cls_name, 0) + 1
                self.counted_ids.add(object_id)
                self.history.append((frame_idx, object_id, cls_name, timestamp))
                return True

        return False

    def get_summary(self):
        """Trả về dict tóm tắt counts và tổng."""
        total = sum(self.counts.values())
        return {"counts": dict(self.counts), "total": total}

    def save_history_csv(self, path="outputs/count_history.csv"):
        """Lưu toàn bộ history hiện tại ra CSV (ghi đè)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "object_id", "class", "timestamp"])
            for rec in self.history:
                writer.writerow(rec)
        return path
