from pathlib import Path

from PIL import Image, ImageFilter
import cog

class Model(cog.Model):

    def setup(self):
        pass

    @cog.input("image", type=Path, help="Input image")
    @cog.input("blur", type=float, help="Blur radius")
    def run(self, image, blur):
        if blur == 0:
            return image
        im = Image.open(str(image))
        im = im.filter(ImageFilter.BoxBlur(blur))
        out_path = cog.make_temp_path("out.png")
        im.save(str(out_path))
        return out_path
