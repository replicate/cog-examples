import os
os.environ["TORCH_HOME"] = "."

from cog import BasePredictor, Input, Path
from PIL import Image
from torchvision import models

WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V1


class Predictor(BasePredictor):
    def setup(self):
        self.model = models.resnet50(weights=WEIGHTS)
        self.model.eval()

    def predict(self, image: Path = Input(description="Image to classify")) -> dict:
        img = Image.open(image).convert("RGB")
        preds = self.model(WEIGHTS.transforms()(img).unsqueeze(0))
        top3 = preds[0].softmax(0).topk(3)
        categories = WEIGHTS.meta["categories"]
        return {categories[i]: float(p) for p, i in zip(*top3)}
