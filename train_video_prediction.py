import getpass
import json
import os
import uuid
from typing import List

import pytorch_lightning.metrics.functional as PL_F
import torch
import torch.nn.functional as F
import torchvision.datasets.utils
import torchvision.transforms.functional as TV_F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from project.callbacks.checkpoint import SaveCheckpointAtEpochEnd
from project.model.model import SelfSupervisedVideoPredictionModel
from project.utils.info import print_device, seed
from project.utils.train import custom_collate

print_device()
seed(42)

username = getpass.getuser()
config = json.load(open(f"{username}.json"))
CLASSIFICATION_DATASET = config["classification"]["root"]
CHECKPOINT_PATH = config["prediction"]["model"]


# Dataset:
# https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
class SelfSupervisedVideoPredictionLitModel(LightningModule):
    def __init__(
        self,
        hidden_dims: List[int],
        image_dim: int = 224,
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
        self.image_dim = image_dim

        self.save_hyperparameters()

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

        self.data_transforms = {
            "video": transforms.Compose(
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
        }

        self.model = SelfSupervisedVideoPredictionModel(
            hidden_dims=hidden_dims,
            latent_block_dims=[self.image_dim // v for v in (2, 4, 8, 16)],
        )

        self.classes = list(sorted(datasets.utils.list_dir(config["ucf101"]["root"])))
        self.class_to_idx = {i: self.classes[i] for i in range(len(self.classes))}

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
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        train_dataset = datasets.UCF101(
            config["ucf101"]["root"],
            config["ucf101"]["anno"],
            frames_per_clip=6,
            step_between_clips=12,
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
        # pick 5 frames, first 3 are seeds, then predict next 3
        curr = x[:, :5, :, :, :]
        # curr = F.pad(curr, [0] * 7 + [5 - curr.shape[1]], "constant", 0)
        curr = curr.reshape(-1, 3, self.image_dim, self.image_dim).contiguous()
        pred = self(curr)
        next = x[:, -3:, :, :, :]
        # next = F.pad(next, [0] * 7 + [5 - next.shape[1]], "constant", 0)
        next = next.reshape(-1, 3, self.image_dim, self.image_dim).contiguous()
        next = F.max_pool2d(next, 2)
        loss = self.criterion(pred, next)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # Save Hidden Layers under batch numbers
        x, y = batch
        curr = x[:, :5, :, :, :]
        curr = curr.reshape(-1, 3, self.image_dim, self.image_dim).contiguous()
        convgru_hidden_states = self.model.forward(curr, test=True)
        dim = convgru_hidden_states.shape[0]
        convgru_hidden_states = torch.unbind(convgru_hidden_states)
        for i in range(dim):
            torch.save(
                torch.squeeze(convgru_hidden_states[i]),
                f"{CLASSIFICATION_DATASET}/{self.class_to_idx[y[i].item()]}/{uuid.uuid1()}.pt",
            )


checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_PATH, save_top_k=1, verbose=True, monitor="epoch", mode="max",
)


def train_model(
    lit_model: SelfSupervisedVideoPredictionLitModel,
    tensorboard_graph_name: str = None,
):
    logger = False
    if tensorboard_graph_name:
        logger = TensorBoardLogger("lightning_logs", name=tensorboard_graph_name)
    # profiler = AdvancedProfiler()

    if os.path.exists(CHECKPOINT_PATH):
        trainer = Trainer(
            resume_from_checkpoint=CHECKPOINT_PATH,
            checkpoint_callback=checkpoint_callback,
            callbacks=[SaveCheckpointAtEpochEnd(filepath=CHECKPOINT_PATH)],
            val_check_interval=0.5,
            logger=logger,
            gpus=1,
            num_nodes=1,
            deterministic=True,
            max_epochs=config["prediction"]["epochs"],
            # limit_train_batches=0.001,
        )
    else:
        trainer = Trainer(
            # precision=16, # 2x speedup but NAN loss after 500 steps
            # profiler=profiler,
            # max_steps=100,  # for profiler
            checkpoint_callback=checkpoint_callback,
            callbacks=[SaveCheckpointAtEpochEnd(filepath=CHECKPOINT_PATH)],
            val_check_interval=0.5,
            logger=logger,
            gpus=1,
            num_nodes=1,
            deterministic=True,
            max_epochs=config["prediction"]["epochs"],
        )
    trainer.fit(lit_model)
    return lit_model, trainer


lit_model = SelfSupervisedVideoPredictionLitModel(
    hidden_dims=[64, 64, 128, 256], batch_size=8
)

lit_model, trainer = train_model(lit_model, "video_prediction")

print("Completed Video Prediction Training")
