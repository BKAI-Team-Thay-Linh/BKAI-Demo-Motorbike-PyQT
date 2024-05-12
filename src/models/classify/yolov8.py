import os
import sys

from PIL import Image
from ultralytics import YOLO

sys.path.append(os.getcwd())


class YoloV8Classifier:
    def __init__(self, device: str = "cuda"):
        self.device = device

    def load_weight(self, weight_path: str):
        self.model = YOLO(weight_path).to(self.device)

    def classify(self, frame) -> int:
        """
        Classify objects in the frame.

        Args:
            frame (np.array): The frame to classify.

        Returns:
            tuple: The bounding boxes, scores, and class IDs (x1, y1, x2, y2, score, class_id)

        """

        result = self.model.predict(frame, verbose=False)[0]
        return result.probs.top1


if __name__ == "__main__":
    yolo = YoloV8Classifier()
    yolo.load_weight("weight/classify/YoloV8.pt")

    image = "assets/bbox/img.png"
    img = Image.open(image)
    print(yolo.classify(img))
