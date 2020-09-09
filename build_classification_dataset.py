import getpass
import json
import os
import shutil
import sys

from pytorch_lightning import Trainer

from project.train_video_prediction import SelfSupervisedVideoPredictionLitModel
from project.utils.cli import query_yes_no
from project.utils.info import print_device, seed

print_device()
seed(42)

username = getpass.getuser()
config = json.load(open(f"{username}.json"))
CLASSIFICATION_DATASET = config["classification"]["root"]
CHECKPOINT_PATH = config["prediction"]["model"]


def load_model(lit_model: SelfSupervisedVideoPredictionLitModel, ):
    if os.path.exists(CHECKPOINT_PATH):
        trainer = Trainer(
            resume_from_checkpoint=CHECKPOINT_PATH,
            checkpoint_callback=False,
            logger=False,
            gpus=1,
            num_nodes=1,
            deterministic=True,
            # limit_test_batches=0.01, # for testing
            max_epochs=config["prediction"]["epochs"],
        )
        trainer.fit(lit_model)
        return lit_model, trainer
    else:
        print("Error! Cannot load checkpoint at {}")
        sys.exit(-1)


lit_model = SelfSupervisedVideoPredictionLitModel(
    hidden_dims=[64, 64, 128, 256], batch_size=8
)

lit_model, trainer = load_model(lit_model)


def build_dataset(dataset_save_dir, dataloader):
    os.makedirs(dataset_save_dir, exist_ok=True)
    os.makedirs(CLASSIFICATION_DATASET, exist_ok=True)
    if len(os.listdir(dataset_save_dir)) != 0:
        if not query_yes_no(
                f"You have a populated classification dataset at {dataset_save_dir}. "
                "Do you want to and regenerate it (The existing folder will be deleted)?"
        ):
            print("Choosing to keep the existing dataset.")
            return
    shutil.rmtree(dataset_save_dir)
    shutil.rmtree(CLASSIFICATION_DATASET)
    os.makedirs(CLASSIFICATION_DATASET, exist_ok=True)
    for folder in lit_model.classes:
        os.mkdir(f"{CLASSIFICATION_DATASET}/{folder}")
    trainer.test(ckpt_path=CHECKPOINT_PATH, test_dataloaders=dataloader)
    os.rename(CLASSIFICATION_DATASET, dataset_save_dir)


build_dataset(
    dataset_save_dir=os.path.abspath(os.path.join(CLASSIFICATION_DATASET, "../train")),
    dataloader=lit_model.train_dataloader(),
)
build_dataset(
    dataset_save_dir=os.path.abspath(os.path.join(CLASSIFICATION_DATASET, "../test")),
    dataloader=lit_model.test_dataloader(),
)

print("Finished Building Dataset")
