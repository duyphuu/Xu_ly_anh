from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")   # tự tải nếu chưa có
print("✅ YOLO model loaded successfully!")

img = np.zeros((640, 640, 3), dtype=np.uint8)
results = model.predict(img, verbose=False)
print("✅ Predict OK. Number of detections:", len(results))
