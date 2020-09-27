import os

import pytorch_lightning.metrics.functional as PL_F
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets.utils
import torchvision.datasets.utils
import torchvision.transforms.functional as TV_F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

from project.callbacks.checkpoint import SaveCheckpointAtEpochEnd
from project.config.user_config import (
    PREDICTION_MODEL_CHECKPOINT,
    PREDICTION_MAX_EPOCHS,
)
from project.config.user_config import (
    UCF101_ANNO_PATH,
    UCF101_WORKERS,
    DATALOADER_WORKERS,
)
from project.config.user_config import UCF101_ROOT_PATH
from project.model.model import SelfSupervisedVideoPredictionModel
from project.transforms.video import (
    random_augment_video_frames,
    RandomFrameRate,
    RestrictFrameRate,
)
from project.utils.function import get_kwargs
from project.utils.train import custom_collate
from project.utils.train import double_resolution


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


class SelfSupervisedVideoPredictionLitModel(LightningModule):
    def __init__(
        self,
        batch_size,
        image_dim: int = 224,
        lr: float = 0.0015,
        l1_loss_wt: float = 0.30,
        l2_loss_wt: float = 0.45,
        ssim_loss_wt: float = 0.25,
        freeze_epochs=3,
    ):
        super().__init__()
        # self.datamodule = datamodule
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

        # self.image_dim = image_dim
        self.fold = 1

        self.train_transforms = transforms.Compose(
            [
                # adjust frames
                RandomFrameRate(p=0.3, in_len=12, out_len=6),
                # scale in [0, 1] of type float
                transforms.Lambda(lambda x: x / 255.0),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(order_video_image_dimensions),
                # normalize
                transforms.Lambda(normalize_video_images),
                # augment video frames
                transforms.Lambda(random_augment_video_frames),
                # # for half precision training
                # transforms.Lambda(lambda x: x.half()),
            ]
        )

        self.test_transforms = transforms.Compose(
            [
                # adjust frames
                RestrictFrameRate(out_len=6),
                # scale in [0, 1] of type float
                transforms.Lambda(lambda x: x / 255.0),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(order_video_image_dimensions),
                # normalize
                transforms.Lambda(normalize_video_images),
                # for half precision training
                # transforms.Lambda(lambda x: x.half()),
            ]
        )

        self.classes = list(sorted(datasets.utils.list_dir(UCF101_ROOT_PATH)))
        self.class_to_idx = {i: self.classes[i] for i in range(len(self.classes))}

        # self.classes = datamodule.classes
        # self.class_to_idx = datamodule.class_to_idx

    def criterion(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        ssim_loss = (1.0 - PL_F.ssim(t1, t2, data_range=1.0)) / 2.0
        l1_loss = F.l1_loss(t1, t2)
        l2_loss = F.mse_loss(t1, t2)

        return (
            self.ssim_loss_wt * ssim_loss
            + self.l1_loss_wt * l1_loss
            + self.l2_loss_wt * l2_loss
        )

    def forward(self, x, seq_len):
        x = self.model.forward(x, seq_len=seq_len)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        return [optimizer], [scheduler]

    def predict_frame(self, batch, batch_nb):
        x, y = batch
        x = x[:, :6, :, :, :]
        # pick 5 frames, first 3 are seeds, then predict next 3
        seed_frames = x[:, :3, :, :, :]
        curr = seed_frames
        b, t, c, w, h = curr.shape
        num_new_frames = 2
        for i in range(num_new_frames):
            seq_len = t + i
            curr = curr.reshape(-1, 3, self.image_dim, self.image_dim).contiguous()
            # print("C0", curr.shape)
            pred = self.forward(curr, seq_len)
            # print("P", pred.shape)
            # print("seq", seq_len)
            pred_frame_only = double_resolution(pred[seq_len - 1 :: seq_len, :, :, :])

            # print("PFO", pred_frame_only.shape)
            curr = torch.cat(
                [
                    curr.view(b, seq_len, c, w, h),
                    pred_frame_only.reshape(b, 1, c, w, h),
                ],
                dim=1,
            ).detach_()
            # print("C1", curr.shape)
        # print("CR", curr.shape)
        curr = curr.reshape(-1, 3, self.image_dim, self.image_dim)
        # print("CR", curr.shape)
        pred = self.forward(curr, seq_len=5)
        # print("PR", pred.shape)
        pred = pred.view(-1, 5, *pred.shape[-3:])
        # print("PR", pred.shape)
        pred = pred[:, -3:, :, :, :].reshape(-1, *pred.shape[-3:]).contiguous()
        # print("PR", pred.shape)

        # curr = F.pad(curr, [0] * 7 + [5 - curr.shape[1]], "constant", 0)
        # curr = curr.reshape(-1, 3, self.image_dim, self.image_dim).contiguous()
        # pred = self(curr)
        next = x[:, -3:, :, :, :]
        # next = F.pad(next, [0] * 7 + [5 - next.shape[1]], "constant", 0)
        next = next.reshape(-1, 3, self.image_dim, self.image_dim).contiguous()
        next = F.max_pool2d(next, 2)
        loss = self.criterion(pred, next)

        return curr, next, pred, loss

    def train_dataloader(self):
        print("Train Dataloader Called")
        dataset = datasets.UCF101(
            UCF101_ROOT_PATH,
            UCF101_ANNO_PATH,
            frames_per_clip=12,
            step_between_clips=100,
            num_workers=UCF101_WORKERS,
            train=True,
            transform=self.train_transforms,
            fold=self.fold,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            collate_fn=custom_collate,
            shuffle=True,
            # pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        print("Val Dataloader Called")
        dataset = datasets.UCF101(
            UCF101_ROOT_PATH,
            UCF101_ANNO_PATH,
            frames_per_clip=12,
            step_between_clips=100,
            num_workers=UCF101_WORKERS,
            train=False,
            transform=self.train_transforms,
            fold=self.fold,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            collate_fn=custom_collate,
            shuffle=True,
            # pin_memory=True,
        )
        return loader

    def training_step(self, batch, batch_nb):
        curr, next, pred, loss = self.predict_frame(batch, batch_nb)

        if batch_nb % 100 == 0:
            self.logger.experiment.add_image(
                "curr",
                torchvision.utils.make_grid(curr, normalize=True),
                self.global_step,
            )
            self.logger.experiment.add_image(
                "next",
                torchvision.utils.make_grid(next, normalize=True),
                self.global_step,
            )
            self.logger.experiment.add_image(
                "pred",
                torchvision.utils.make_grid(pred, normalize=True),
                self.global_step,
            )

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        curr, next, pred, loss = self.predict_frame(batch, batch_nb)
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
                    SaveCheckpointAtEpochEnd(filepath=PREDICTION_MODEL_CHECKPOINT)
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
    # ucf101_dm = UCF101VideoDataModule(batch_size=1)
    # for validation dataloader
    # ucf101_dm.setup("test")
    lit_model = SelfSupervisedVideoPredictionLitModel(batch_size=4)
    lit_model, trainer = load_or_train_model(
        lit_model,
        tensorboard_graph_name="prediction-3-recursive-small-p",
        # resume=False,
        # save=False,
        # validation=False,
    )
    print("Completed Video Prediction Training")
