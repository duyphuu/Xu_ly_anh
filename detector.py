# detector.py
# Wrapper nhỏ cho YOLOv8 (ultralytics)
from ultralytics import YOLO
import numpy

# Các nhãn COCO mà ta quan tâm (car, motorcycle, bus, truck)
VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.model = YOLO(model_path)
        # ultralytics đặt device trong .predict bằng param device nếu cần.

    def detect(self, frame, conf=0.25, iou=0.45):
        """
        Trả về list các detections: mỗi detection = dict {bbox, conf, cls_name, cls_id, xyxy}
        bbox ở dạng [x1,y1,x2,y2]
        """
        results = self.model.predict(source=frame, conf=conf, iou=iou, verbose=False)
        r = results[0]
        detections = []

        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()  # [N,4]
        scores = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        # map class id -> name using model.names
        names = r.names if hasattr(r, "names") else self.model.names

        for bb, sc, cid in zip(boxes, scores, cls_ids):
            cls_name = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
            if cls_name in VEHICLE_CLASS_NAMES:
                detections.append({
                    "bbox": bb.astype(int).tolist(),
                    "conf": float(sc),
                    "cls_id": int(cid),
                    "cls_name": cls_name
                })
        return detections
