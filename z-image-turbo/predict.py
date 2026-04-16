import os
import tempfile

import torch
from cog import BasePredictor, Path
from diffusers import ZImagePipeline

WEIGHTS_DIR = "/src/weights"


class Predictor(BasePredictor):
    def setup(self):
        if not os.path.isdir(WEIGHTS_DIR):
            raise FileNotFoundError(
                f"Weights directory not found: {WEIGHTS_DIR}. "
                "Managed weights must be downloaded and mounted before the container starts."
            )

        model_index = os.path.join(WEIGHTS_DIR, "model_index.json")
        if not os.path.exists(model_index):
            raise FileNotFoundError(
                f"model_index.json not found in {WEIGHTS_DIR}. "
                "Weights directory exists but appears incomplete."
            )

        self.model = ZImagePipeline.from_pretrained(
            WEIGHTS_DIR,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        self.model.to("cuda")

    def predict(self, prompt: str) -> Path:
        image = self.model(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,  # This actually results in 8 DiT forwards
            guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        output_path = Path(tempfile.mktemp(suffix=".png"))
        image.save(output_path)
        return output_path
