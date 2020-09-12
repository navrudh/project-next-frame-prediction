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


# Dataset:
# https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
class UCF101VideoDataModule(LightningDataModule):
    def __init__(self, batch_size=1, image_dim=224):
        super().__init__()
        self.batch_size = batch_size
        self.image_dim = image_dim

        def order_video_image_dimensions(x):
            return x.permute(0, 3, 1, 2)

        def normalize_video_images(x):
            for img in x:
                TV_F.normalize(
                    img,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    inplace=True,
                )
            return x

        self.train_transforms = self.test_transforms = transforms.Compose(
            [
                # scale in [0, 1] of type float
                transforms.Lambda(lambda x: x / 255.0),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(order_video_image_dimensions),
                # normalize
                transforms.Lambda(normalize_video_images),
                # rescale to the most common size
                transforms.Lambda(
                    lambda x: F.interpolate(x, (self.image_dim, self.image_dim))
                ),
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
                frames_per_clip=6,
                step_between_clips=12,
                num_workers=UCF101_WORKERS,
                train=True,
                transform=self.train_transforms,
            )

        if stage == "test" or stage is None:
            self.test_dataset = datasets.UCF101(
                UCF101_ROOT_PATH,
                UCF101_ANNO_PATH,
                frames_per_clip=6,
                step_between_clips=12,
                num_workers=UCF101_WORKERS,
                train=False,
                transform=self.test_transforms,
            )

        # # train/val split
        # mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # # assign to use in dataloaders
        # self.train_dataset = mnist_train
        # self.val_dataset = mnist_val
        # self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            collate_fn=custom_collate,
            shuffle=True,
        )

    def test_dataloader(self):
        # shuffling because we want the first few random gifs generated
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            collate_fn=custom_collate,
        )
