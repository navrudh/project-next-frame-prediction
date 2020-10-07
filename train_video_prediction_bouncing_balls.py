import os

import pytorch_lightning.metrics.functional as PL_F
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets.utils
import torchvision.datasets.utils
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from project.callbacks.checkpoint import SaveCheckpointAtEpochEnd
from project.config.user_config import (
    DATALOADER_WORKERS,
    BB_SIZE,
    BB_TIMESTEPS,
    BB_NBALLS,
)
from project.config.user_config import (
    PREDICTION_MODEL_CHECKPOINT,
    PREDICTION_MAX_EPOCHS,
    WORK_DIR,
)
from project.config.user_config import UCF101_ROOT_PATH
from project.dataset.bouncing_balls import BouncingBalls
from project.model.model import SelfSupervisedVideoPredictionModel
from project.transforms.video import augment_bouncing_balls_video_frames
from project.utils.function import get_kwargs
from project.utils.train import collate_bouncing_balls, double_resolution


def order_video_image_dimensions(x):
    return x.permute(0, 3, 1, 2)


class BouncingBallsVideoPredictionLitModel(LightningModule):
    def __init__(
        self,
        batch_size,
        image_dim: int = 224,
        lr: float = 0.0001,
        l1_loss_wt: float = 0.16,
        l2_loss_wt: float = 0.05,
        ssim_loss_wt: float = 0.84,
        freeze_epochs=3,
    ):
        super().__init__()
        self.batch_size = batch_size
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

        self.fold = 1
        self.val_dataset = None
        self.train_dataset = None

        self.classes = list(sorted(datasets.utils.list_dir(UCF101_ROOT_PATH)))
        self.class_to_idx = {i: self.classes[i] for i in range(len(self.classes))}

    def criterion(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        ssim_loss = 1.0 - PL_F.ssim(t1, t2, data_range=1.0)
        l1_loss = F.l1_loss(t1, t2)
        l2_loss = F.mse_loss(t1, t2)

        return ssim_loss + l1_loss + l2_loss

    def forward(self, x, hidden):
        x, hidden = self.model.forward(x, hidden=hidden)
        return x, hidden

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        return [optimizer], [scheduler]

    def predict_frame(self, batch, batch_nb):
        x = batch
        # pick 5 frames, first 3 are seeds, then predict next 3
        b, t, c, w, h = x.shape
        n_predicted = 3
        t -= n_predicted

        input_frames = x[:, :t, :, :, :]
        predicted_frames = []

        hidden = None
        for i in range(n_predicted):
            pred, hidden = self.forward(input_frames, hidden)
            last_predicted_frame = pred.view(b, t, *pred.shape[-3:])[:, -1:, :, :, :]

            if i < (n_predicted - 1):
                last_predicted_frame = last_predicted_frame
                upscaled_pred_frame = double_resolution(
                    last_predicted_frame.view(-1, *last_predicted_frame.shape[-3:])
                )
                input_frames = torch.cat(
                    [
                        input_frames[:, : t - 1, :, :, :],
                        upscaled_pred_frame.reshape(b, 1, c, w, h),
                    ],
                    dim=1,
                ).detach_()

            predicted_frames.append(last_predicted_frame)

        pred3 = torch.cat(predicted_frames, dim=1)
        pred3 = pred3.view(-1, *pred3.shape[-3:])

        inp = x[:, :6, :, :, :]
        inp = inp.view(-1, 3, self.image_dim, self.image_dim)
        inp = F.max_pool2d(inp, 2)

        next = inp.view(-1, 6, *inp.shape[-3:])[:, -3:, :, :, :]
        next = next.reshape(-1, *next.shape[-3:]).contiguous()
        loss = self.criterion(pred3, next)

        pred6 = inp.view(-1, 6, *inp.shape[-3:]).detach().clone()
        pred6[:, -3:, :, :, :] = pred3.view(-1, 3, *pred3.shape[-3:])
        pred6 = pred6.view(-1, *pred6.shape[-3:])

        return inp, pred6, loss

    def dataset_init(self, stage):
        if stage == "train" and not self.train_dataset:
            self.train_dataset = BouncingBalls(
                size=BB_SIZE,
                timesteps=BB_TIMESTEPS,
                n_balls=BB_NBALLS,
                mode="train",
                train_size=2000,
                transform=transforms.Lambda(augment_bouncing_balls_video_frames),
            )

        if stage == "val" and not self.val_dataset:
            self.val_dataset = BouncingBalls(
                size=BB_SIZE,
                timesteps=BB_TIMESTEPS,
                n_balls=BB_NBALLS,
                mode="val",
                train_size=2000,
            )

    def train_dataloader(self):
        print("Train Dataloader Called")
        self.dataset_init("train")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        print("Val Dataloader Called")
        self.dataset_init("val")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        inp, pred, loss = self.predict_frame(batch, batch_nb)

        if batch_nb % 100 == 0:
            self.logger.experiment.add_image(
                "input",
                torchvision.utils.make_grid(inp, normalize=True, nrow=6),
                self.global_step,
            )
            self.logger.experiment.add_image(
                "pred",
                torchvision.utils.make_grid(pred, normalize=True, nrow=6),
                self.global_step,
            )

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        inp, pred, loss = self.predict_frame(batch, batch_nb)
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

lr_logger = LearningRateLogger(logging_interval="epoch")


def load_or_train_model(
    lit_model: BouncingBallsVideoPredictionLitModel,
    tensorboard_graph_name: str = None,
    save=True,
    resume=True,
    validation=True,
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

    kwargs.update(
        get_kwargs(
            logger=logger,
            gpus=1,
            # deterministic=True,
            max_epochs=PREDICTION_MAX_EPOCHS,
            callbacks=[lr_logger],
            # limit_train_batches=0.001,
            # limit_val_batches=0.1,
            # val_check_interval=0.5,
        )
    )

    if save:
        kwargs.update(
            get_kwargs(
                checkpoint_callback=checkpoint_callback,
                callbacks=[
                    *kwargs["callbacks"],
                    SaveCheckpointAtEpochEnd(filepath=PREDICTION_MODEL_CHECKPOINT),
                ],
            )
        )
    if gif_mode:
        kwargs.update(get_kwargs(limit_test_batches=0.001, limit_train_batches=0.001))

    if not validation:
        kwargs.update(
            get_kwargs(
                limit_val_batches=0.0, val_check_interval=0.0, num_sanity_val_steps=0
            )
        )

    if resume and os.path.exists(PREDICTION_MODEL_CHECKPOINT):
        print("Found existing model at ", PREDICTION_MODEL_CHECKPOINT)
        trainer = Trainer(resume_from_checkpoint=PREDICTION_MODEL_CHECKPOINT, **kwargs)
        print("Resuming training ...")
    else:
        print("Begin training from scratch ...")
        trainer = Trainer(
            # precision=16, # 2x speedup but NAN loss after 500 steps
            # profiler=profiler,
            # max_steps=100,  # for profiler
            **kwargs,
        )
    trainer.fit(lit_model)
    return lit_model, trainer


if __name__ == "__main__":
    # ucf101_dm = UCF101VideoDataModule(batch_size=1)
    # for validation dataloader
    # ucf101_dm.setup("test")
    lit_model = BouncingBallsVideoPredictionLitModel(batch_size=4)
    lit_model, trainer = load_or_train_model(
        lit_model,
        tensorboard_graph_name=WORK_DIR.split("/")[-1],
        # resume=False,
        # save=False,
        # validation=False,
    )
    print("Completed Video Prediction Training")