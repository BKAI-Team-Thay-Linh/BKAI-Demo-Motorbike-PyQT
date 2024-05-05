import os
import sys

sys.path.append(os.getcwd())

from ultralytics import YOLO


class YoloDectector:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = YOLO(model_path).to(device)

    def detect(self, conf: float, frame) -> tuple:
        """
        Detect objects in the frame.

        Args:
            conf (float): Confidence threshold.
            frame (

        Returns:
            tuple: The bounding boxes, scores, and class IDs (x1, y1, x2, y2, score, class_id)
        """
        results = self.model.predict(frame, verbose=False, conf=conf)[0]
        bboxes_coor = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
        return bboxes_coor, scores, class_ids
