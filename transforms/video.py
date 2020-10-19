import random
from functools import partial
from typing import Union, List

import torch
import torchvision
from pytorch_lightning import seed_everything
from torch import Tensor

from config.user_config import PREDICTION_MODEL_H


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
        self.out_len = out_len

    def forward(self, x: Union[Tensor, List[Tensor]]):
        """
        Args:
            x (Tensor or List of Tensors): input frames.

        Returns:
            Tensor or List of Tensors: selected frames.
        """
        if torch.rand(1) < self.p:
            choice = self.choices[0]
        else:
            choice = self.choices[-1]

        return x[choice]

    def __repr__(self):
        return self.__class__.__name__ + "(p={}, out_len={})".format(
            self.p, self.out_len
        )


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
        return self.__class__.__name__ + "(out_len={})".format(self.out_len)


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, p=0.5, mean=0, var=0.05):
        super().__init__()
        self.p = p
        self.mean = mean
        self.var = var

    def __call__(self, tensor):
        return torch.clamp(
            tensor + torch.randn_like(tensor[0]) * self.var + self.mean, min=0, max=1
        )

    def __repr__(self):
        return self.__class__.__name__ + "(p={}, mean={}, var={})".format(
            self.p, self.var, self.mean
        )


ucf101_video_augmentation = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomVerticalFlip(0.1),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        ),
        torchvision.transforms.RandomChoice(
            [
                torchvision.transforms.Resize((PREDICTION_MODEL_H, PREDICTION_MODEL_H)),
                torchvision.transforms.RandomResizedCrop(
                    PREDICTION_MODEL_H, scale=(0.33, 1.0)
                ),
                torchvision.transforms.RandomCrop(PREDICTION_MODEL_H),
            ]
        ),
        torchvision.transforms.ToTensor(),
    ]
)

bb_video_augmentation = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomVerticalFlip(0.1),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
        ),
        torchvision.transforms.Resize((PREDICTION_MODEL_H, PREDICTION_MODEL_H)),
        torchvision.transforms.ToTensor(),
    ]
)


def seed_and_call(seed, func, *args, **kwargs):
    seed_everything(seed)
    return func(*args, **kwargs)


def augment_video_frames(augmentation, x):
    seed = random.randint(0, 1000000)

    return torch.stack([seed_and_call(seed, augmentation, img) for img in x])


augment_ucf101_video_frames = partial(augment_video_frames, ucf101_video_augmentation)
augment_bouncing_balls_video_frames = partial(
    augment_video_frames, bb_video_augmentation
)
