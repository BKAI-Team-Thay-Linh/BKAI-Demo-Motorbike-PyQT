import os
import sys

sys.path.append(os.getcwd())

from ultralytics import YOLO


class YoloDectector:
    def __init__(
        self, model_path: str, device: str = "cuda", will_classify: bool = False
    ):
        self.model = YOLO(model_path).to(device)
        self.will_classify = will_classify

    def detect(self, conf: float, frame, **kwargs) -> tuple:
        """
        Detect objects in the frame.

        Args:
            conf (float): Confidence threshold.
            frame (

        Returns:
            tuple: The bounding boxes, scores, and class IDs (x1, y1, x2, y2, score, class_id)
        """
        if kwargs.get("yolo_class_ids"):
            yolo_class_ids: list = kwargs.get("yolo_class_ids")
            results = self.model.predict(
                frame, verbose=False, conf=conf, classes=yolo_class_ids
            )[0]
        else:
            results = self.model.predict(frame, verbose=False, conf=conf)[0]
        bboxes_coor = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        if self.will_classify:
            class_ids = results.boxes.cls.cpu().numpy()
            return bboxes_coor, scores, class_ids
        return bboxes_coor, scores
