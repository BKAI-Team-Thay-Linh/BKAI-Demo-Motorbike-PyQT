import torch
import torch.nn as nn

# Load model directly
from transformers import AutoModelForImageClassification


class ViTBase(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(ViTBase, self).__init__()

        self.model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.model(x).logits


if __name__ == "__main__":
    model = ViTBase()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
