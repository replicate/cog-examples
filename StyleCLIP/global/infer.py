from pathlib import Path
import os
import shutil
import numpy as np
import torch
import clip
from PIL import Image
import pickle
import copy
from MapTS import GetFs, GetBoundary, GetDt
from manipulate import Manipulator

import cog


class Model(cog.Model):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.fs3 = np.load("./npy/ffhq/fs3.npy")
        np.set_printoptions(suppress=True)

    @cog.input(
        "img_index", type=int, min=0, max=1999, help="Image index from FFHQ dataset"
    )
    @cog.input("neutral", type=str, help="Neutral image description")
    @cog.input("target", type=str, help="Target image description")
    @cog.input(
        "manipulation_strength",
        type=float,
        min=-10,
        max=10,
        default=3,
        help="The higher the manipulation strength, the closer the generated image becomes to the target description. Negative values moves the generated image further from the target description",
    )
    @cog.input(
        "disentanglement_threshold",
        type=float,
        min=0.08,
        max=0.3,
        default=0.2,
        help="The higher the disentanglement threshold, the more specific the changes are to the target attribute. Lower values mean that broader changes are made to the input image",
    )
    def run(
        self,
        img_index,
        neutral,
        target,
        manipulation_strength,
        disentanglement_threshold,
    ):
        M = Manipulator(dataset_name="ffhq")
        img_indexs = [img_index]
        dlatent_tmp = [tmp[img_indexs] for tmp in M.dlatents]
        M.num_images = len(img_indexs)

        M.alpha = [0]
        M.manipulate_layers = [0]
        codes, out = M.EditOneC(0, dlatent_tmp)
        original = Image.fromarray(out[0, 0]).resize((512, 512))
        M.manipulate_layers = None
        classnames = [target, neutral]
        dt = GetDt(classnames, self.model)

        M.alpha = [manipulation_strength]
        boundary_tmp2, _ = GetBoundary(
            self.fs3, dt, M, threshold=disentanglement_threshold
        )
        codes = M.MSCode(dlatent_tmp, boundary_tmp2)
        out = M.GenerateImg(codes)
        generated = Image.fromarray(out[0, 0])  # .resize((512,512))

        temp_dir = cog.make_temp_dir()
        original.save(os.path.join(temp_dir, "original.jpg"))
        generated.save(os.path.join(temp_dir, "generated.jpg"))
        archive_path = shutil.make_archive("images", "zip", temp_dir)
        return Path(archive_path)
