import os

import pytorch_lightning.metrics.functional as PL_F
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from project.callbacks.checkpoint import SaveCheckpointAtEpochEnd
from project.config.user_config import (
    PREDICTION_MODEL_CHECKPOINT,
    PREDICTION_MAX_EPOCHS,
)
from project.dataset.ucf101video import UCF101VideoDataModule
from project.model.model import SelfSupervisedVideoPredictionModel
from project.utils.function import get_kwargs
from project.utils.info import print_device, seed

print_device()
seed(42)


class SelfSupervisedVideoPredictionLitModel(LightningModule):
    def __init__(
        self,
        datamodule,
        image_dim: int = 224,
        lr: float = 0.0015,
        l1_loss_wt: float = 0.35,
        l2_loss_wt: float = 0.50,
        ssim_loss_wt: float = 0.15,
        freeze_epochs=3,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.lr = lr
        self.l1_loss_wt = l1_loss_wt
        self.l2_loss_wt = l2_loss_wt
        self.ssim_loss_wt = ssim_loss_wt
        self.image_dim = image_dim
        self.freeze_epochs = freeze_epochs

        self.save_hyperparameters()

        self.model = SelfSupervisedVideoPredictionModel(
            latent_block_dims=[self.image_dim // v for v in (2, 4, 8, 16)],
        )
        self.model.freeze_encoder()

        self.classes = datamodule.classes
        self.class_to_idx = datamodule.class_to_idx

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
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x[:, :6, :, :, :]
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

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x = x[:, :6, :, :, :]
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
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def on_epoch_start(self):
        if self.current_epoch == self.freeze_epochs:
            self.model.unfreeze_encoder()


checkpoint_callback = ModelCheckpoint(
    filepath=PREDICTION_MODEL_CHECKPOINT,
    save_top_k=1,
    verbose=True,
    monitor="epoch",
    mode="max",
)


def load_or_train_model(
    lit_model: SelfSupervisedVideoPredictionLitModel,
    tensorboard_graph_name: str = None,
    save=True,
    gif_mode=False,
    profiler=None,
):
    logger = False
    if profiler:
        trainer = Trainer(
            logger=logger,
            profiler=profiler,
            gpus=1,
            max_steps=150,
            limit_val_batches=0.1,
        )
        trainer.fit(lit_model)
        return lit_model, trainer

    if tensorboard_graph_name:
        logger = TensorBoardLogger("lightning_logs", name=tensorboard_graph_name)
    # profiler = AdvancedProfiler()

    kwargs = {}
    if save:
        kwargs.update(
            get_kwargs(
                checkpoint_callback=checkpoint_callback,
                callbacks=[
                    SaveCheckpointAtEpochEnd(filepath=PREDICTION_MODEL_CHECKPOINT)
                ],
            )
        )
    if gif_mode:
        kwargs.update(get_kwargs(limit_test_batches=0.001, limit_train_batches=0.001))

    kwargs.update(
        get_kwargs(
            logger=logger,
            gpus=1,
            deterministic=True,
            max_epochs=PREDICTION_MAX_EPOCHS,
            # limit_train_batches=0.001,
            limit_val_batches=0.1,
            # val_check_interval=0.5,
        )
    )

    if os.path.exists(PREDICTION_MODEL_CHECKPOINT):
        trainer = Trainer(resume_from_checkpoint=PREDICTION_MODEL_CHECKPOINT, **kwargs)
    else:
        trainer = Trainer(
            # precision=16, # 2x speedup but NAN loss after 500 steps
            # profiler=profiler,
            # max_steps=100,  # for profiler
            **kwargs,
        )
    trainer.fit(lit_model)
    return lit_model, trainer


if __name__ == "__main__":
    ucf101_dm = UCF101VideoDataModule(batch_size=4)
    # for validation dataloader
    ucf101_dm.setup("test")
    lit_model = SelfSupervisedVideoPredictionLitModel(datamodule=ucf101_dm)
    lit_model, trainer = load_or_train_model(
        lit_model, tensorboard_graph_name="video_prediction",
    )
    print("Completed Video Prediction Training")
