import imageio
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TV_F
from PIL import Image, ImageOps

from config.user_config import PREDICTION_MODEL_H

PREDICTION_COLOR = (34, 139, 34)  # GREEN
SOURCE_COLOR = (200, 0, 4)  # RED


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def generate_gif(pil_inps, pil_preds, gif_name, pred_start_idx=3, border_px=4):
    image_list = []
    for idx, (inp, pred) in enumerate(zip(pil_inps, pil_preds)):
        inp_image_w_border = ImageOps.expand(inp, border=border_px, fill=SOURCE_COLOR)
        pred_image_w_border = ImageOps.expand(
            pred,
            border=border_px,
            fill=PREDICTION_COLOR if idx >= pred_start_idx else SOURCE_COLOR,
        )
        image_list.append(
            np.array(get_concat_h(inp_image_w_border, pred_image_w_border))
        )

    imageio.mimwrite(gif_name, image_list, format="GIF", duration=0.75)


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
