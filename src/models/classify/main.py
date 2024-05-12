import os
import sys

import numpy as np

sys.path.append(os.getcwd())  # NOQA

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.models import models_logger
from src.models.classify.resnet18 import ResNet18
from src.models.classify.vit import ViTBase
from src.models.classify.yolov8 import YoloV8Classifier


class Transform:
    def __init__(self):
        self.transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])

    def __call__(self, image):
        return self.transform(image=image)["image"]


class Models(torch.nn.Module):
    def __init__(self, model: str = "resnet18", num_classes: int = 3):
        super(Models, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model
        if model == "resnet18":
            self.model = ResNet18(num_classes=num_classes).to(self.device)
        elif model == "vit":
            self.model = ViTBase(num_classes=num_classes).to(self.device)
        elif model == "yolov8":
            self.model = YoloV8Classifier(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.eval()

    def forward(self, x):
        return self.model(x)

    def load_weight(self, weight_path: str):
        if not self.model_type == "yolov8":
            checkpoint = torch.load(weight_path, map_location=self.device)
            self.load_state_dict(checkpoint["state_dict"], strict=False)
            models_logger.info(f"Weight has been loaded from {weight_path}")
        else:
            self.model.load_weight(weight_path)
            models_logger.info(f"Weight has been loaded from {weight_path}")

    def infer(self, image: Image) -> int:
        if not self.model_type == "yolov8":
            img_np = np.array(image.convert("RGB"))
            img = Transform()(img_np).to(self.device)

            with torch.no_grad():
                pred = self(img.unsqueeze(0))

            return torch.argmax(pred, dim=1).item()
        else:
            return self.model.classify(image)

    @property
    def name(self):
        return self.model.__class__.__name__


if __name__ == "__main__":
    model = Models(model="vit")
    model.load_weight("weight/classify/ViT.ckpt")

    output = []

    for img_name in os.listdir(".temp/extracted_frames"):
        img_path = ".temp/Screenshot 2024-05-07 194615.png"
        img = Image.open(img_path)

        result = model.infer(img)

        print(f"Image: {img_name}, Prediction: {result}")

        output.append({"image": img_name, "prediction": result})
