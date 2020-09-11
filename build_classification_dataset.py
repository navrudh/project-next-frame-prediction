import os
import shutil
import sys

from pytorch_lightning import Trainer
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets

from project.config.user_config import (
    UCF101_ROOT_PATH,
    UCF101_ANNO_PATH,
    UCF101_WORKERS,
    DATALOADER_WORKERS,
    PREDICTION_MODEL_CHECKPOINT,
    CLASSIFICATION_DATASET_PATH,
    PREDICTION_MAX_EPOCHS,
)
from project.train_video_prediction import SelfSupervisedVideoPredictionLitModel
from project.utils.cli import query_yes_no
from project.utils.info import print_device, seed
from project.utils.train import custom_collate

print_device()
seed(42)


class ClassificationDatasetBuilder(SelfSupervisedVideoPredictionLitModel):
    def test_dataloader(self):
        test_dataset = datasets.UCF101(
            UCF101_ROOT_PATH,
            UCF101_ANNO_PATH,
            frames_per_clip=6,
            step_between_clips=8,
            num_workers=UCF101_WORKERS,
            train=False,
            transform=self.data_transforms["video"],
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            shuffle=True,
            collate_fn=custom_collate,
        )
        return test_dataloader


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


lit_model = ClassificationDatasetBuilder(hidden_dims=[64, 64, 128, 256], batch_size=8)

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


build_dataset(
    dataset_save_dir=os.path.abspath(
        os.path.join(CLASSIFICATION_DATASET_PATH, "../train")
    ),
    dataloader=lit_model.train_dataloader(),
)
build_dataset(
    dataset_save_dir=os.path.abspath(
        os.path.join(CLASSIFICATION_DATASET_PATH, "../test")
    ),
    dataloader=lit_model.test_dataloader(),
)

print("Finished Building Dataset")
