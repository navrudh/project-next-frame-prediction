import getpass
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import device as torch_device, cuda as torch_cuda
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import DatasetFolder

device = torch_device("cuda:0" if torch_cuda.is_available() else "cpu")
print("Device:", device)
if torch_cuda.is_available():
    print("Device Name:", torch_cuda.get_device_name(0))

seed = 42
seed_everything(seed)
print("Set Random Seed:", seed)

username = getpass.getuser()
config = json.load(open(f"{username}.json"))
CLASSIFICATION_DATASET = config["classification"]["root"]


class VideoClassificationModel(LightningModule):
    def __init__(self, batch_size=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.model = nn.Sequential(nn.Linear(768, 101))
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
            os.path.abspath(os.path.join(CLASSIFICATION_DATASET, "../train")),
            loader=torch.load,
            extensions=tuple(".pt"),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,)
        return dataloader

    def val_dataloader(self):
        dataset = DatasetFolder(
            os.path.abspath(os.path.join(CLASSIFICATION_DATASET, "../test")),
            loader=torch.load,
            extensions=tuple(".pt"),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size,)
        return dataloader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).detach() / (len(y) * 1.0)

        tensorboard_logs = {"train_loss": loss, "train_acc": acc}
        tqdm_dict = {"acc": acc}
        return {"loss": loss, "progress_bar": tqdm_dict, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).detach() / (len(y) * 1.0)

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
        num_nodes=1,
        deterministic=True,
        max_epochs=max_epochs,
    )
    trainer.fit(lit_model)
    return lit_model, trainer


lit_model = VideoClassificationModel(batch_size=256)
lit_model, trainer = train_model(lit_model, "video_classification")

print("Finished Classification")
