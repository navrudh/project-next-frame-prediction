import random
from typing import Union, List

import torch
import torchvision
from pytorch_lightning import seed_everything
from torch import Tensor


class RandomFrameRate(torch.nn.Module):
    """Pick frames equally distant from each other as per the out_len

    Args:
        p (float): probability of varying framerate. Default value is 0.5
        in_len (int): number of input frames. Default value is 12
        out_len (int): number of output frames. Default value is 6
    """

    def __init__(self, p=0.5, in_len=12, out_len=6):
        super().__init__()
        self.p = p
        self.choices = [
            torch.tensor([skip_frame * i for i in range(out_len)])
            for skip_frame in range(1, 1 + in_len // out_len)
        ]

    def forward(self, x: Union[Tensor, List[Tensor]]):
        """
        Args:
            x (Tensor or List of Tensors): input frames.

        Returns:
            Tensor or List of Tensors: selected frames.
        """
        if torch.rand(1) < self.p:
            choice = self.choices[-1]
        else:
            choice = self.choices[0]

        return x[choice]

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RestrictFrameRate(torch.nn.Module):
    """Adjust video framerate

    Args:
        p (float): probability of varying framerate. Default value is 0.5
    """

    def __init__(self, out_len=6):
        super().__init__()
        self.out_len = out_len

    def forward(self, x: Union[Tensor, List[Tensor]]):
        """
        Args:
            x (Tensor or List of Tensors): input frames.

        Returns:
            Tensor or List of Tensors: selected frames.
        """

        return x[: self.out_len]

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


_image_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomVerticalFlip(0.1),
        torchvision.transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25
        ),
        torchvision.transforms.RandomChoice(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomRotation(45),
                        torchvision.transforms.CenterCrop(224),
                    ]
                ),
                torchvision.transforms.RandomResizedCrop(224, scale=(0.33, 1.0)),
                torchvision.transforms.RandomCrop(224),
            ]
        ),
        torchvision.transforms.ToTensor(),
    ]
)


def seed_and_call(seed, func, *args, **kwargs):
    seed_everything(seed)
    return func(*args, **kwargs)


def random_augment_video_frames(x):
    seed = random.randint(0, 1000000)

    return [seed_and_call(seed, _image_transforms, img) for img in x]
