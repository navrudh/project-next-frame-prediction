import torch.nn.functional as F
import torchvision.datasets.utils
import torchvision.transforms.functional as TV_F
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from project.config.user_config import (
    UCF101_ROOT_PATH,
    UCF101_ANNO_PATH,
    UCF101_WORKERS,
    DATALOADER_WORKERS,
)
from project.utils.train import custom_collate


def order_video_image_dimensions(x):
    return x.permute(0, 3, 1, 2)


def normalize_video_images(x):
    for img in x:
        TV_F.normalize(
            img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True,
        )
    return x


def unnormalize_video_images(x):
    for img in x:
        TV_F.normalize(
            img,
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            inplace=True,
        )
    return x


invert_transforms = transforms.Compose([transforms.Lambda(unnormalize_video_images)])


# Dataset:
# https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
class UCF101VideoDataModule(LightningDataModule):
    def __init__(self, batch_size=1, image_dim=224, fold=1):
        super().__init__()
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.fold = fold

        self.train_transforms = self.test_transforms = transforms.Compose(
            [
                # scale in [0, 1] of type float
                transforms.Lambda(lambda x: x / 255.0),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(order_video_image_dimensions),
                # normalize
                # transforms.Lambda(normalize_video_images),
                # rescale to the most common size
                transforms.Lambda(lambda x: F.interpolate(x, (224, 224))),
                # transforms.Lambda(lambda x: x.half()),
            ]
        )

        self.classes = list(sorted(datasets.utils.list_dir(UCF101_ROOT_PATH)))
        self.class_to_idx = {i: self.classes[i] for i in range(len(self.classes))}

    def setup(self, stage=None):
        # transform

        if stage == "fit" or stage is None:
            self.train_dataset = datasets.UCF101(
                UCF101_ROOT_PATH,
                UCF101_ANNO_PATH,
                frames_per_clip=12,
                step_between_clips=100,
                num_workers=UCF101_WORKERS,
                train=True,
                transform=self.train_transforms,
                fold=self.fold,
            )

        if stage == "test" or stage is None:
            self.test_dataset = datasets.UCF101(
                UCF101_ROOT_PATH,
                UCF101_ANNO_PATH,
                frames_per_clip=6,
                step_between_clips=100,
                num_workers=UCF101_WORKERS,
                train=False,
                transform=self.test_transforms,
                fold=self.fold,
            )

    def train_dataloader(self):
        print("Train Dataloader Called")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            collate_fn=custom_collate,
            shuffle=True,
            # pin_memory=True,
        )

    def val_dataloader(self):
        print("Val Dataloader Called")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            collate_fn=custom_collate,
            shuffle=True,
            # pin_memory=True,
        )

    def test_dataloader(self):
        print("Test Dataloader Called")
        # shuffling because we want the first few random gifs generated
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            collate_fn=custom_collate,
            shuffle=True,
            # pin_memory=True,
        )
