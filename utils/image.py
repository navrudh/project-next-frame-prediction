from pathlib import Path

import imageio
import torch.nn.functional as F
import torchvision.transforms.functional as TV_F

from config.user_config import PREDICTION_MODEL_H


def generate_gif(image_dir, file_glob, gif_name):
    image_path = Path(image_dir)
    images = sorted(list(image_path.glob(file_glob)))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))

    imageio.mimwrite(gif_name, image_list, format="GIF", duration=1)


def unnormalize_video_images(x):
    for img in x:
        TV_F.normalize(
            img,
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            inplace=True,
        )
    return x


def image_int_to_float(x):
    return x / 255.0


def rescale_tensor(x):
    return F.interpolate(x, (PREDICTION_MODEL_H, PREDICTION_MODEL_H))


def order_video_image_dimensions(x):
    return x.permute(0, 3, 1, 2)
