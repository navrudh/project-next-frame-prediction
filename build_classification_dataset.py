import getpass
import json
import os
import shutil
import sys

from pytorch_lightning import Trainer, seed_everything
from torch import device as torch_device, cuda as torch_cuda

from project.train_video_prediction import SelfSupervisedVideoPredictionLitModel
from project.utils.cli import query_yes_no

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
CHECKPOINT_PATH = config["prediction"]["model"]


def load_model(lit_model: SelfSupervisedVideoPredictionLitModel,):
    if os.path.exists(CHECKPOINT_PATH):
        trainer = Trainer(
            resume_from_checkpoint=CHECKPOINT_PATH,
            checkpoint_callback=False,
            logger=False,
            gpus=1,
            num_nodes=1,
            deterministic=True,
            limit_train_batches=0.001,
            limit_test_batches=0.1,
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

os.makedirs(CLASSIFICATION_DATASET, exist_ok=True)
if len(os.listdir()) != 0:
    if not query_yes_no(
        f"You have a populated classification dataset at {CLASSIFICATION_DATASET}. "
        "Do you want to and regenerate it (The existing folder will be deleted)?"
    ):
        print("Choosing to keep the existing dataset.")
        exit()

shutil.rmtree(CLASSIFICATION_DATASET)
os.makedirs(CLASSIFICATION_DATASET)
for folder in lit_model.classes:
    os.mkdir(f"{CLASSIFICATION_DATASET}/{folder}")
trainer.test(ckpt_path=CHECKPOINT_PATH)

print("Finished Building Dataset")
