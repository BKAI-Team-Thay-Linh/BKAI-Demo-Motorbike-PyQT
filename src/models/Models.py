import os
import sys

sys.path.append(os.getcwd())  # NOQA

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.models import models_logger
from src.models.ResNet18 import ResNet18


class Transform:
    def __init__(self):
        self.transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])

    def __call__(self, image):
        return self.transform(image=image)["image"]


class Models(torch.nn.Module):
    def __init__(self, model: str = "resnet18", num_classes: int = 3):
        super(Models, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        models_logger.info(f"Using device: {self.device}")

        if model == "resnet18":
            self.model = ResNet18(num_classes=num_classes).to(self.device)
        self.eval()

    def forward(self, x):
        return self.model(x)

    def load_weight(self, weight_path: str):
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"], strict=False)

        models_logger.info(f"Weight has been loaded from {weight_path}")

    def infer(self, image: Image) -> int:
        img_np = np.array(image.convert("RGB"))
        img = Transform()(img_np).to(self.device)

        with torch.no_grad():
            pred = self(img.unsqueeze(0))

        return torch.argmax(pred, dim=1).item()


if __name__ == "__main__":
    model = Models()
    model.load_weight("weight/resnet18.ckpt")

    for img_name in os.listdir(".temp/cropped_bboxes"):
        img_path = f".temp/cropped_bboxes/{img_name}"
        img = Image.open(img_path)
        print(model.infer(img))
