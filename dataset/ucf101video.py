import torchvision.datasets.utils
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from config.user_config import (
    UCF101_ROOT_PATH,
    UCF101_ANNO_PATH,
    UCF101_WORKERS,
    DATALOADER_WORKERS,
    UCF101_SBC,
)
from utils.image import image_int_to_float, order_video_image_dimensions, rescale_tensor
from utils.train import collate_ucf101


# Dataset:
# https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
class UCF101VideoDataModule(LightningDataModule):
    def __init__(self, batch_size=1, image_dim=224, fold=1):
        super().__init__()
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.fold = fold

        self.test_transforms = transforms.Compose(
            [
                # adjust frames
                # RestrictFrameRate(out_len=6),
                # scale in [0, 1] of type float
                transforms.Lambda(image_int_to_float),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(order_video_image_dimensions),
                # # normalize
                # transforms.Lambda(normalize_video_images),
                # rescale to the most common size
                transforms.Lambda(rescale_tensor),
                # for half precision training
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
                step_between_clips=UCF101_SBC,
                num_workers=UCF101_WORKERS,
                train=True,
                transform=self.test_transforms,
                fold=self.fold,
            )

        if stage == "test" or stage is None:
            self.test_dataset = datasets.UCF101(
                UCF101_ROOT_PATH,
                UCF101_ANNO_PATH,
                frames_per_clip=6,
                step_between_clips=UCF101_SBC,
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
            collate_fn=collate_ucf101,
            shuffle=True,
            # pin_memory=True,
        )

    def val_dataloader(self):
        print("Val Dataloader Called")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            collate_fn=collate_ucf101,
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
            collate_fn=collate_ucf101,
            shuffle=True,
            # pin_memory=True,
        )
