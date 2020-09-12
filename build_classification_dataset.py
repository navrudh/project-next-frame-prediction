import os
import shutil
import sys
import uuid

import torch
from pytorch_lightning import Trainer

from project.config.user_config import (
    PREDICTION_MODEL_CHECKPOINT,
    CLASSIFICATION_DATASET_PATH,
    PREDICTION_MAX_EPOCHS,
)
from project.dataset.ucf101video import UCF101VideoDataModule
from project.train_video_prediction import SelfSupervisedVideoPredictionLitModel
from project.utils.cli import query_yes_no
from project.utils.info import print_device, seed

print_device()
seed(42)


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


def load_model(lit_model: SelfSupervisedVideoPredictionLitModel,):
    if os.path.exists(PREDICTION_MODEL_CHECKPOINT):
        trainer = Trainer(
            resume_from_checkpoint=PREDICTION_MODEL_CHECKPOINT,
            checkpoint_callback=False,
            logger=False,
            gpus=1,
            num_nodes=1,
            deterministic=True,
            # limit_test_batches=0.01, # for testing
            max_epochs=PREDICTION_MAX_EPOCHS,
        )
        trainer.fit(lit_model)
        return lit_model, trainer
    else:
        print("Error! Cannot load checkpoint at {}")
        sys.exit(-1)


ucf101_dm = UCF101VideoDataModule(batch_size=8)
lit_model = ClassificationDatasetBuilder(datamodule=ucf101_dm)

lit_model, trainer = load_model(lit_model)


def build_dataset(dataset_save_dir, dataloader):
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
    for folder in lit_model.classes:
        os.mkdir(f"{CLASSIFICATION_DATASET_PATH}/{folder}")
    trainer.test(ckpt_path=PREDICTION_MODEL_CHECKPOINT, test_dataloaders=dataloader)
    os.rename(CLASSIFICATION_DATASET_PATH, dataset_save_dir)


ucf101_dm.setup()
build_dataset(
    dataset_save_dir=os.path.abspath(
        os.path.join(CLASSIFICATION_DATASET_PATH, "../train")
    ),
    dataloader=ucf101_dm.train_dataloader(),
)
build_dataset(
    dataset_save_dir=os.path.abspath(
        os.path.join(CLASSIFICATION_DATASET_PATH, "../test")
    ),
    dataloader=ucf101_dm.test_dataloader(),
)

print("Finished Building Dataset")
