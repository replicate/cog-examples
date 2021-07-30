import tempfile
from pathlib import Path

from PIL import Image, ImageFilter
import cog


class Predictor(cog.Predictor):
    def setup(self):
        pass

    @cog.input("input", type=Path, help="Input image")
    @cog.input("blur", type=float, help="Blur radius", default=5)
    def predict(self, input, blur):
        if blur == 0:
            return input
        im = Image.open(str(input))
        im = im.filter(ImageFilter.BoxBlur(blur))
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        im.save(str(out_path))
        return out_path
