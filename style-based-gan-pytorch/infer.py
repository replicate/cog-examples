import math

import cog
import torch
from torchvision import utils

from generate import sample, get_mean_style
from model import StyledGenerator


SIZE = 1024


class Model(cog.Model):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = StyledGenerator(512).to(self.device)
        self.generator.load_state_dict(
            torch.load(
                "stylegan-1024px-new.model",
                map_location=self.device,
            )["g_running"],
        )
        self.generator.eval()

    def run(self):
        mean_style = get_mean_style(self.generator, self.device)
        step = int(math.log(SIZE, 2)) - 2
        img = sample(self.generator, step, mean_style, 1, self.device)
        output_path = cog.make_temp_path("output.png")
        utils.save_image(img, output_path, normalize=True)
        return output_path
