import cog
from pathlib import Path
from models.colorize import Colorizer

class InstColorizationModel(cog.Model):
    def setup(self):
        self.colorizer = Colorizer()

    @cog.input("input", type=Path, help="grayscale input image")
    def run(self, input):
        output_path = self.colorizer.colorize(input)
        return output_path

