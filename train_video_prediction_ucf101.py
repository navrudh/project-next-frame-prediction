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

from callbacks.checkpoint import SaveCheckpointAtEpochEnd
from config.user_config import (
    PREDICTION_MODEL_CHECKPOINT,
    WORK_DIR,
    save_config,
    SAVE_CFG_KEY_DATASET,
    PREDICTION_TRAINER_KWARGS,
    PREDICTION_BATCH_SIZE,
    PREDICTION_LR,
    PREDICTION_DECAY,
    PREDICTION_PATIENCE,
    PREDICTION_SCHED_FACTOR,
    PREDICTION_MODEL_H,
    PREDICTION_RESNET_UNFREEZE_EPOCH,
    PREDICTION_MOMENTUM,
)
from config.user_config import (
    UCF101_ANNO_PATH,
    UCF101_WORKERS,
    DATALOADER_WORKERS,
)
from config.user_config import UCF101_ROOT_PATH
from model.model import SelfSupervisedVideoPredictionModel
from transforms.video import (
    RandomFrameRate,
    augment_ucf101_video_frames,
)
from utils.function import get_kwargs
from utils.train import collate_ucf101


def order_video_image_dimensions(x):
    return x.permute(0, 3, 1, 2)


def image_int_to_float(x):
    return x / 255.0


def rescale_tensor(x):
    return F.interpolate(x, (PREDICTION_MODEL_H, PREDICTION_MODEL_H))


class UCF101VideoPredictionLitModel(LightningModule):
    def __init__(
            self,
            batch_size,
            image_dim: int = 224,
            lr: float = 0.0001,
            wt_decay=1e-5,
            momentum=0.9,
            sched_patience=3,
            sched_factor=0.1,
            freeze_epochs=3,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.wt_decay = wt_decay
        self.sched_patience = sched_patience
        self.sched_factor = sched_factor
        self.image_dim = image_dim
        self.freeze_epochs = freeze_epochs
        self.momentum = momentum

        self.save_hyperparameters()

        self.model = SelfSupervisedVideoPredictionModel(
            latent_block_dims=[self.image_dim // v for v in (2, 4, 8, 16)],
            seed_frames=3,
        )
        self.model.freeze_encoder()

        self.fold = 1
        self.val_dataset = None
        self.train_dataset = None

        self.train_transforms = transforms.Compose(
            [
                # adjust frames
                RandomFrameRate(p=0.2, in_len=12, out_len=6),
                # scale in [0, 1] of type float
                transforms.Lambda(image_int_to_float),
                # reshape into (T, C, H, W) for easier convolutions
                transforms.Lambda(order_video_image_dimensions),
                # # normalize
                # transforms.Lambda(normalize_video_images),
                # augment video frames
                transforms.Lambda(augment_ucf101_video_frames),
                # # for half precision training
                # transforms.Lambda(lambda x: x.half()),
            ]
        )

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

    def criterion(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        ssim_loss = 1.0 - PL_F.ssim(t1, t2, data_range=1.0)
        l1_loss = F.l1_loss(t1, t2)
        l2_loss = F.mse_loss(t1, t2)

        return ssim_loss + l1_loss + l2_loss

    def forward(self, x):
        x, hidden = self.model.forward(x)
        return x, hidden

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wt_decay,
            momentum=self.hparams.momentum,
        )
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lr_lambda=lambda epoch: 0.0001 * (0.5 ** (epoch / 100)),
        # )
        return optimizer

    def predict_frame(self, batch, batch_nb):
        x, y = batch
        # pick 6 frames, first 3 are seeds (defined in model), last 3 are targets (calculated below)
        x = x[:, :6, :, :, :]
        b, t, c, w, h = x.shape

        gradX = x
        pred, hidden = self.forward(gradX)

        loss = self.criterion(
            x[:, self.model.seed_frames:].detach().reshape(-1, c, w, h),
            pred[:, self.model.seed_frames:].reshape(-1, c, w, h),
        )

        inp = x.reshape(-1, c, w, h)
        pred = pred.reshape(-1, c, w, h)
        return inp, pred, loss

    def dataset_init(self, stage):
        if stage == "train" and not self.train_dataset:
            self.train_dataset = datasets.UCF101(
                UCF101_ROOT_PATH,
                UCF101_ANNO_PATH,
                frames_per_clip=12,
                step_between_clips=75,
                num_workers=UCF101_WORKERS,
                train=True,
                transform=self.train_transforms,
                fold=self.fold,
            )

        if stage == "val" and not self.val_dataset:
            self.val_dataset = datasets.UCF101(
                UCF101_ROOT_PATH,
                UCF101_ANNO_PATH,
                frames_per_clip=6,
                step_between_clips=75,
                num_workers=UCF101_WORKERS,
                train=False,
                transform=self.test_transforms,
                fold=self.fold,
            )

    def train_dataloader(self):
        print("Train Dataloader Called")
        self.dataset_init("train")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            collate_fn=collate_ucf101,
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
            collate_fn=collate_ucf101,
            shuffle=False,
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        inp, pred, loss = self.predict_frame(batch, batch_nb)
        if batch_nb == 0:
            self.logger.experiment.add_image(
                "input",
                torchvision.utils.make_grid(inp, normalize=True, nrow=6),
                self.current_epoch,
            )
            self.logger.experiment.add_image(
                "pred",
                torchvision.utils.make_grid(pred, normalize=True, nrow=6),
                self.current_epoch,
            )
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        inp, pred, loss = self.predict_frame(batch, batch_nb)
        if batch_nb == 0:
            self.logger.experiment.add_image(
                "val_pred",
                torchvision.utils.make_grid(pred, normalize=True, nrow=6),
                self.current_epoch,
            )
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        tqdm_dict = {"val_loss": avg_loss}
        return {
            "avg_val_loss": avg_loss,
            "progress_bar": tqdm_dict,
            "log": tensorboard_logs,
        }

    def on_epoch_start(self):
        if self.current_epoch == self.freeze_epochs:
            print("Unfreezing Resnet")
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
        lit_model: LightningModule,
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

    # Override with User Settings
    kwargs.update(PREDICTION_TRAINER_KWARGS)

    if resume and os.path.exists(PREDICTION_MODEL_CHECKPOINT):
        print("Found existing model at ", PREDICTION_MODEL_CHECKPOINT)
        trainer = Trainer(
            resume_from_checkpoint=PREDICTION_MODEL_CHECKPOINT,
            # precision=16,
            **kwargs,
        )
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
    additional_config = {SAVE_CFG_KEY_DATASET: "ucf101"}
    save_config(additional_config)
    lit_model = UCF101VideoPredictionLitModel(
        image_dim=PREDICTION_MODEL_H,
        batch_size=PREDICTION_BATCH_SIZE,
        lr=PREDICTION_LR,
        wt_decay=PREDICTION_DECAY,
        momentum=PREDICTION_MOMENTUM,
        sched_patience=PREDICTION_PATIENCE,
        sched_factor=PREDICTION_SCHED_FACTOR,
        freeze_epochs=PREDICTION_RESNET_UNFREEZE_EPOCH,
    )
    lit_model, trainer = load_or_train_model(
        lit_model,
        tensorboard_graph_name=WORK_DIR.split("/")[-1],
        # resume=False,
        # save=False,
        # validation=False,
    )
    print("Completed Video Prediction Training")
