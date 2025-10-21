# video_io.py
# Xử lý video: đọc, detect, track, count, hiển thị và lưu CSV + output video

import cv2
import pandas as pd
import time
import os
from detector import VehicleDetector
from tracker import CentroidTracker
from counter import Counter

def process_video(input_path, output_path="outputs/output.mp4", csv_path="outputs/counts.csv",
                  model_path="yolov8n.pt", display=False):
    """
    Xử lý 1 video: detect, track, đếm xe, lưu kết quả ra video + CSV.
    Trả về dict gồm: đường dẫn video, CSV và counts tổng.
    """

    # --- Chuẩn bị ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không thể mở video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector = VehicleDetector(model_path=model_path)
    tracker = CentroidTracker(max_disappeared=40, max_distance=60)

    # --- Đặt line đếm ---
    line_y = height // 2
    counter = Counter(line_position_y=line_y, direction="down")

    prev_centroids = {}  # id -> centroid
    frame_idx = 0
    start = time.time()

    # --- Vòng lặp đọc từng frame ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = detector.detect(frame, conf=0.35)
        rects = []
        cls_map = {}  # bbox tuple -> class name
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            rects.append([x1, y1, x2, y2])
            cls_map[(x1, y1, x2, y2)] = d["cls_name"]

        tracked = tracker.update(rects)  # id -> bbox

        # --- Vẽ và đếm ---
        for oid, bbox in tracked.items():
            x1, y1, x2, y2 = bbox
            cX, cY = int((x1 + x2)/2), int((y1 + y2)/2)
            curr_centroid = (cX, cY)
            cls_name = cls_map.get(tuple(bbox), "car")  # fallback nếu không tìm thấy class

            prev_cent = prev_centroids.get(oid, curr_centroid)
            counted = counter.check_and_count(oid, prev_cent, curr_centroid, cls_name, frame_idx)

            color = (0, 255, 0) if not counted else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cX, cY), 3, color, -1)
            cv2.putText(frame, f"ID {oid} {cls_name}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            prev_centroids[oid] = curr_centroid

        # --- Hiển thị line + thống kê ---
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
        status_text = "Counts: " + ", ".join([f"{k}:{v}" for k, v in counter.counts.items()])
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        if display:
            cv2.imshow("Result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    print(f"✅ Processed {frame_idx} frames in {elapsed:.2f}s, FPS ~ {frame_idx/elapsed:.2f}")

    # --- Lưu CSV kết quả ---
    df = pd.DataFrame(counter.history, columns=["frame", "object_id", "class", "timestamp"])
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV: {csv_path}")
    print(f"💾 Saved output video: {output_path}")

    return {"video": output_path, "csv": csv_path, "counts": counter.counts}
