import getpass
import json
from typing import List

import pytorch_lightning.metrics.functional as PL_F
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, device as torch_device, cuda as torch_cuda
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from project.model.model import SelfSupervisedVideoPredictionModel

device = torch_device("cuda:0" if torch_cuda.is_available() else "cpu")
print("Device:", device)
if torch_cuda.is_available():
    print("Device Name:", torch_cuda.get_device_name(0))

seed = 42
seed_everything(seed)
print("Set Random Seed:", seed)

username = getpass.getuser()
config = json.load(open(f"{username}.json"))


def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


# Dataset:
# https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
# Annotations:
# HHHHHHHHHHH
class SelfSupervisedVideoPredictionLitModel(LightningModule):
    def __init__(
        self,
        hidden_dims: List[int],
        latent_block_dims: List[int],
        batch_size: int = 1,
        l1_loss_wt: int = 0.3,
        l2_loss_wt: int = 0.05,
        ssim_loss_wt: int = 0.65,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.l1_loss_wt = l1_loss_wt
        self.l2_loss_wt = l2_loss_wt
        self.ssim_loss_wt = ssim_loss_wt

        self.save_hyperparameters()

        self.data_transforms = {
            "video": transforms.Compose(
                [
                    # scale in [0, 1] of type float
                    transforms.Lambda(lambda x: x / 255.0),
                    # reshape into (T, C, H, W) for easier convolutions
                    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                    # rescale to the most common size
                    transforms.Lambda(
                        lambda x: nn.functional.interpolate(x, (224, 224))
                    ),
                    # transforms.Lambda(lambda x: x.half()),
                ]
            )
        }

        self.model = SelfSupervisedVideoPredictionModel(
            hidden_dims=hidden_dims,
            latent_block_dims=latent_block_dims,
            batch_size=batch_size,
        )

    def criterion(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        ssim_loss = 1.0 - PL_F.ssim(t1, t2, data_range=1.0)
        l1_loss = F.l1_loss(t1, t2)
        l2_loss = F.mse_loss(t1, t2)

        return (
            self.ssim_loss_wt * ssim_loss
            + self.l1_loss_wt * l1_loss
            + self.l2_loss_wt * l2_loss
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0005, weight_decay=1e-5
        )
        return optimizer

    def train_dataloader(self) -> DataLoader:
        print("train_dataloader -- loading")
        train_dataset = datasets.UCF101(
            config["ucf101"]["root"],
            config["ucf101"]["anno"],
            frames_per_clip=6,
            step_between_clips=8,
            num_workers=config["ucf101"]["workers"],
            train=True,
            transform=self.data_transforms["video"],
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=config["dataloader"]["workers"],
            shuffle=True,
            collate_fn=custom_collate,
        )
        return train_dataloader

    def training_step(self, batch, batch_nb):
        x, y = batch
        curr = x[:, :5, :, :, :]
        # curr = F.pad(curr, [0] * 7 + [5 - curr.shape[1]], "constant", 0)
        curr = curr.reshape(-1, 3, 224, 224).contiguous()
        pred = self(curr)
        next = x[:, -3:, :, :, :]
        # next = F.pad(next, [0] * 7 + [5 - next.shape[1]], "constant", 0)
        next = next.reshape(-1, 3, 224, 224).contiguous()
        next = F.interpolate(next, size=(112, 112))
        loss = self.criterion(pred, next)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}


def train_model(
    lit_model: LightningModule, tensorboard_graph_name: str = None, max_epochs: int = 1,
):
    logger = False
    if tensorboard_graph_name:
        logger = TensorBoardLogger("lightning_logs", name=tensorboard_graph_name)
    # profiler = AdvancedProfiler()
    trainer = Trainer(
        # precision=16, # 2x speedup but NAN loss after 500 steps
        # profiler=profiler,
        checkpoint_callback=False,
        logger=logger,
        gpus=1,
        num_nodes=1,
        # deterministic=True,
        max_epochs=max_epochs,
        limit_train_batches=0.1,
        # max_steps=100,
        # progress_bar_refresh_rate=0,
        # progress_bar_callback=False,
    )
    trainer.fit(lit_model)
    return lit_model


IMG_DIM = 224
block_inp_dims = [IMG_DIM // v for v in (2, 4, 8, 16)]

lit_model = SelfSupervisedVideoPredictionLitModel(
    hidden_dims=[64, 64, 128, 256], latent_block_dims=block_inp_dims, batch_size=8
)

train_model(lit_model)
