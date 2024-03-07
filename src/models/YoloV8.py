import os
import sys
sys.path.append(os.getcwd())  # NOQA

from ultralytics import YOLO
import torch
import cv2


class YoloV8:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)

    def detect(self, image_path: str):
        results = self.model.predict(image_path, verbose=False)[0]
        bboxes = results.boxes.xywh.cpu().numpy()
        bboxes[:, :2] -= bboxes[:, 2:] / 2  # xywh to xyxy
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        return bboxes, scores, class_ids

    def draw(self, frame, bboxes, scores, class_ids):
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            x, y, w, h = bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            color = (0, 255, 0)  # Green color for bounding box
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Class: {int(class_id)}, Score: {score:.2f}"
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


if __name__ == '__main__':
    video_path = 'assets/test_vid.mp4'
    model_path = 'weight/best.pt'

    yolo = YoloV8(model_path)

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, scores, class_ids = yolo.detect(frame)
        frame_with_bboxes = yolo.draw(frame, bboxes, scores, class_ids)

        cv2.imshow('frame', frame_with_bboxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
