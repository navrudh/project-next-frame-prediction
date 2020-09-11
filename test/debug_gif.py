# import imageio
#
# from pathlib import Path
#
# image_path = Path('/home/navrudh/Projects/Uni/cudavision/project/local_remote_out/out')
# images = sorted(list(image_path.glob('f9ea1f82-f2d1-11ea-a7e9-c5e4e5c2f13a-*.jpg')))
# image_list = []
# for file_name in images:
#     image_list.append(imageio.imread(file_name))
#
# imageio.mimwrite('animated_from_images.gif', image_list, format='GIF', duration=1)
from project.utils.image import generate_gif

generate_gif(
    "/home/navrudh/Projects/Uni/cudavision/project/local_remote_out/out/",
    file_glob="f9ea1f82-f2d1-11ea-a7e9-c5e4e5c2f13a*.jpg",
)
