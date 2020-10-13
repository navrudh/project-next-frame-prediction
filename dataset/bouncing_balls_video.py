from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from config.user_config import (
    BB_TIMESTEPS,
    BB_NBALLS,
    DATALOADER_WORKERS,
    BB_TRAIN_SIZE,
    PREDICTION_MODEL_H,
)
from dataset.bouncing_balls import BouncingBalls


class BouncingBallsVideoDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_dataset = BouncingBalls(
                size=PREDICTION_MODEL_H,
                timesteps=BB_TIMESTEPS,
                n_balls=BB_NBALLS,
                mode="train",
                train_size=BB_TRAIN_SIZE,
            )

        if stage == "test" or stage is None:
            self.val_dataset = BouncingBalls(
                size=PREDICTION_MODEL_H,
                timesteps=BB_TIMESTEPS,
                n_balls=BB_NBALLS,
                mode="val",
                train_size=BB_TRAIN_SIZE,
            )

    def train_dataloader(self):
        print("Train Dataloader Called")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        print("Val Dataloader Called")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        print("Test Dataloader Called")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            pin_memory=True,
        )
