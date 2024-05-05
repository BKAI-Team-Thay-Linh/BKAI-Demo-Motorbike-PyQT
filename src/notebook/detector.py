import os
import sys

import numpy as np

sys.path.append(os.getcwd())  # NOQA


from typing import Optional

import cv2
import torch
from PIL import Image
from ultralytics import YOLO


class Detector:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, weights: str = "yolov8n.pt"):
        self.model = YOLO(model=weights).to(device=self.DEVICE)

    def detect(self, conf: float = 0.5, frame: Optional[Image.Image] = None):
        """
            This function will detect the object in the frame.
            The result is the tuple of
            ```
            (bboxes_coor, scores, class_ids)
            ```

            Where:
            - bboxes_coor: the bounding box coordinates of the detected object,
            which is in the format of (x, y, x, y)

            - scores: the confidence score of the detected object

            - class_ids: the class id of the detected object
        Args:
            conf (float, optional): The confidence score. Defaults to 0.5.
            frame (Optional[Image.Image], optional): The input frame. Defaults to None.
        """

        results = self.model.predict(frame, verbose=False, conf=conf, classes=[3])[0]

        # Get the bounding box coordinates
        bboxes_coor = results.boxes.xyxy.cpu().numpy()
        # print(f"bbox_coor xywh: {bboxes_coor}")

        # Get the confidence score
        scores = results.boxes.conf.cpu().numpy()

        return bboxes_coor, scores

    def classify(self, bboxes, img: np.ndarray):
        img = Image.fromarray(img)

        class_ids = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

            cropped_img = img.crop((x1, y1, x2, y2))
            class_id = self.classifier.infer(cropped_img)
            class_ids.append(class_id)

        return class_ids

    def draw(
        self,
        img: np.ndarray,
        bboxes,
        scores,
        class_ids,
        mask_alpha=0.3,
    ):
        # Convert the image to numpy array
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        classes = ["motorcycle", "scooter", "undetected"]
        height, width = img.shape[:2]
        colors = {0: [172, 47, 117], 1: [192, 67, 251], 2: [195, 103, 9]}

        # Create mask image
        mask_img = img.copy()
        det_img = img.copy()

        size = min([height, width]) * 0.0006
        text_thickness = int(min([height, width]) * 0.001)

        # Draw bounding boxes and labels of detections
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            if class_id <= 2:
                color = colors[class_id]
                x1, y1, x2, y2 = bbox.astype(int)

                # Draw rectangle
                cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

                # Draw fill rectangle in mask image
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
                label = classes[class_id]
                caption = f"{label} Score: {score:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    text=caption,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=size,
                    thickness=text_thickness,
                )
                th = int(th * 1.2)

                cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
                cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
                cv2.putText(
                    det_img,
                    caption,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    size,
                    (255, 255, 255),
                    text_thickness,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    mask_img,
                    caption,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    size,
                    (255, 255, 255),
                    text_thickness,
                    cv2.LINE_AA,
                )

            else:
                continue

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


if __name__ == "__main__":
    video_path = "assets/5min.mp4"
    model_path = "weight/yolo/yolov9_best.pt"

    yolo = Detector(model_path)

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, scores = yolo.detect(frame=frame, conf=0.3)
        class_ids = yolo.classify(bboxes, frame)
        frame_with_bboxes = yolo.draw(frame, bboxes, scores, class_ids)

        cv2.imshow("frame", frame_with_bboxes)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
