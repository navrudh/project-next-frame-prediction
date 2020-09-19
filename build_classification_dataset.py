import os
import shutil
import sys
import uuid

import torch
from pytorch_lightning import Trainer

from project.config.user_config import (
    PREDICTION_MODEL_CHECKPOINT,
    CLASSIFICATION_DATASET_PATH,
)
from project.dataset.ucf101video import UCF101VideoDataModule
from project.train_video_prediction import SelfSupervisedVideoPredictionLitModel
from project.utils.cli import query_yes_no


class ClassificationDatasetBuilder(SelfSupervisedVideoPredictionLitModel):
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


def load_model(dm):
    if os.path.exists(PREDICTION_MODEL_CHECKPOINT):
        return ClassificationDatasetBuilder.load_from_checkpoint(
            PREDICTION_MODEL_CHECKPOINT, datamodule=dm
        )
    else:
        print("Error! Cannot load checkpoint at {}")
        sys.exit(-1)


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
    lit_model = load_model(ucf101_dm)
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
