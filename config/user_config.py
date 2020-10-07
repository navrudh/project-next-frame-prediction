import getpass
import json
import os

username = getpass.getuser()
current_py_file_path = os.path.dirname(os.path.abspath(__file__))
settings_dir = os.path.abspath(
    os.path.join(current_py_file_path, "../resources/config")
)
config = json.load(open(f"{settings_dir}/user-{username}.json"))

## Dataset Config Vars
# UCF101
UCF101_ROOT_PATH = config["ucf101"]["root"]
UCF101_ANNO_PATH = config["ucf101"]["anno"]
UCF101_WORKERS = config["ucf101"]["workers"]
UCF101_CACHE = config["ucf101"]["cache"]

# Bouncing Balls
BB_SIZE = config["bouncing-balls"]["size"]
BB_NBALLS = config["bouncing-balls"]["n-balls"]
BB_TIMESTEPS = config["bouncing-balls"]["timesteps"]

# Dataloader Config Vars
DATALOADER_WORKERS = config["dataloader"]["workers"]

# Work Dir
WORK_DIR = config["workdir"]

# Prediction Config Vars
PREDICTION_MODEL_CHECKPOINT = WORK_DIR + "/model.ckpt"
PREDICTION_OUTPUT_DIR = WORK_DIR + "/generated"
PREDICTION_MAX_EPOCHS = config["prediction"]["epochs"]

# Classification Config Vars
CLASSIFICATION_DATASET_PATH = WORK_DIR + "/classification/tensors"
