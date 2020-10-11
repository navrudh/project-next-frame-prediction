import os
import shutil
import uuid

import torch
from pytorch_lightning import Trainer

from config.user_config import CLASSIFICATION_DATASET_PATH
from dataset.ucf101video import UCF101VideoDataModule
from train_video_prediction_ucf101 import UCF101VideoPredictionLitModel
from utils.cli import query_yes_no
from utils.train import load_model


class ClassificationDatasetBuilder(UCF101VideoPredictionLitModel):
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
                f"{CLASSIFICATION_DATASET_PATH}/{self.class_to_idx[y[i].item()]}/{uuid.uuid1()}.pt",
            )


def build_dataset(model, dataset_save_dir, dataloader):
    os.makedirs(dataset_save_dir, exist_ok=True)
    os.makedirs(CLASSIFICATION_DATASET_PATH, exist_ok=True)
    if len(os.listdir(dataset_save_dir)) != 0:
        if not query_yes_no(
            f"You have a populated classification dataset at {dataset_save_dir}. "
            "Do you want to and regenerate it (The existing folder will be deleted)?"
        ):
            print("Choosing to keep the existing dataset.")
            return
    shutil.rmtree(dataset_save_dir)
    shutil.rmtree(CLASSIFICATION_DATASET_PATH)
    os.makedirs(CLASSIFICATION_DATASET_PATH, exist_ok=True)
    for folder in model.classes:
        os.mkdir(f"{CLASSIFICATION_DATASET_PATH}/{folder}")
    trainer = Trainer(gpus=1)
    model.datamodule = None
    trainer.test(model, test_dataloaders=dataloader)
    os.rename(CLASSIFICATION_DATASET_PATH, dataset_save_dir)


if __name__ == "__main__":
    ucf101_dm = UCF101VideoDataModule(batch_size=8)
    lit_model = load_model(ClassificationDatasetBuilder, batch_size=8)
    lit_model.eval()
    ucf101_dm.setup("fit")
    build_dataset(
        lit_model,
        dataset_save_dir=os.path.abspath(
            os.path.join(CLASSIFICATION_DATASET_PATH, "../train")
        ),
        dataloader=ucf101_dm.train_dataloader(),
    )

    ucf101_dm = UCF101VideoDataModule(batch_size=8)
    ucf101_dm.setup("test")
    build_dataset(
        lit_model,
        dataset_save_dir=os.path.abspath(
            os.path.join(CLASSIFICATION_DATASET_PATH, "../test")
        ),
        dataloader=ucf101_dm.test_dataloader(),
    )

    print("Finished Building Dataset")
