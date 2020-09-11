from pathlib import Path

import imageio


def generate_gif(image_dir, file_glob, gif_name):
    image_path = Path(image_dir)
    images = sorted(list(image_path.glob(file_glob)))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))

    imageio.mimwrite(gif_name, image_list, format="GIF", duration=1)
