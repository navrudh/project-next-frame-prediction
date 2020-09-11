from pathlib import Path

import imageio


def generate_gif(path, file_glob):
    image_path = Path(path)
    images = sorted(list(image_path.glob(file_glob)))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))

    imageio.mimwrite(f"{path}.gif", image_list, format="GIF", duration=1)
