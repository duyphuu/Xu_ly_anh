# tracker.py
# Một CentroidTracker đơn giản (gán ID theo khoảng cách centroid)
import numpy as np
from scipy.spatial import distance

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # id -> centroid
        self.bboxes = {}   # id -> last bbox
        self.disappeared = {}  # id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.next_object_id += 1
        return oid

    def deregister(self, oid):
        del self.objects[oid]
        del self.bboxes[oid]
        del self.disappeared[oid]

    def update(self, rects):
        """
        rects: list of bbox [x1,y1,x2,y2]
        returns dict id -> bbox
        """
        if len(rects) == 0:
            # tăng counter disappeared cho tất cả
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.bboxes.copy()

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(tuple(input_centroids[i]), rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = distance.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                oid = object_ids[row]
                self.objects[oid] = tuple(input_centroids[col])
                self.bboxes[oid] = rects[col]
                self.disappeared[oid] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # disappeared
            for row in unused_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            # new objects
            for col in unused_cols:
                self.register(tuple(input_centroids[col]), rects[col])

        return self.bboxes.copy()
