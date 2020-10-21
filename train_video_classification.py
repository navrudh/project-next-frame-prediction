import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import DatasetFolder

from config.user_config import CLASSIFICATION_DATASET_PATH, WORK_DIR, DATALOADER_WORKERS


class VideoClassificationModel(LightningModule):
    def __init__(self, batch_size=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.model = nn.Sequential(nn.Linear(768, 2048), nn.Linear(2048, 101))
        self.loss = nn.NLLLoss()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return F.log_softmax(x, -1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = DatasetFolder(
            os.path.abspath(os.path.join(CLASSIFICATION_DATASET_PATH, "../train")),
            loader=torch.load,
            extensions=tuple(".pt"),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=DATALOADER_WORKERS,
        )
        return dataloader

    def val_dataloader(self):
        dataset = DatasetFolder(
            os.path.abspath(os.path.join(CLASSIFICATION_DATASET_PATH, "../test")),
            loader=torch.load,
            extensions=tuple(".pt"),
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=DATALOADER_WORKERS
        )
        return dataloader

    def forward_compute(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).detach() / (len(y) * 1.0)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward_compute(batch, batch_idx)
        tensorboard_logs = {"train_loss": loss, "train_acc": acc}
        tqdm_dict = {"acc": acc}
        return {"loss": loss, "progress_bar": tqdm_dict, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward_compute(batch, batch_idx)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "val_acc": avg_acc}
        tqdm_dict = {"val_loss": avg_loss, "val_acc": avg_acc}
        return {
            "val_loss": avg_loss,
            "progress_bar": tqdm_dict,
            "log": tensorboard_logs,
        }


def train_model(
    lit_model: VideoClassificationModel,
    tensorboard_graph_name: str = None,
    max_epochs: int = 50,
):
    logger = False
    if tensorboard_graph_name:
        logger = TensorBoardLogger("lightning_logs", name=tensorboard_graph_name)
    # profiler = AdvancedProfiler()

    trainer = Trainer(
        # precision=16, # 2x speedup but NAN loss after 500 steps
        # profiler=profiler,
        logger=logger,
        gpus=1,
        deterministic=True,
        max_epochs=max_epochs,
    )
    trainer.fit(lit_model)
    return lit_model, trainer


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    lit_model = VideoClassificationModel(batch_size=256)
    lit_model, trainer = train_model(
        lit_model, tensorboard_graph_name=WORK_DIR.split("/")[-1] + "classification"
    )

    print("Finished Classification")
