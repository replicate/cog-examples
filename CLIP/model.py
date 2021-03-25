import cog
from pathlib import Path
from PIL import Image
import torch

import clip

class CLIP(cog.Model):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net, self.preprocess = clip.load("ViT-B/32", device=self.device)

    @cog.input("image", type=Path)
    @cog.input("labels", type=str, help="comma separated labels")
    def run(self, image, labels):
        processed_image = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)
        processed_text = clip.tokenize(labels.split(",")).to(self.device)

        with torch.no_grad():
            image_features = self.net.encode_image(processed_image)
            text_features = self.net.encode_text(processed_text)
            
            logits_per_image, logits_per_text = self.net(processed_image, processed_text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()
            return {labels.split(",")[i]: prob for i, prob in enumerate(probs[0])}




