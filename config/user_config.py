import getpass
import json
import os

username = getpass.getuser()
current_py_file_path = os.path.dirname(os.path.abspath(__file__))
settings_dir = os.path.abspath(
    os.path.join(current_py_file_path, "../resources/config")
)
config = json.load(open(f"{settings_dir}/user-{username}.json"))

# Dataset Config Vars
UCF101_ROOT_PATH = config["ucf101"]["root"]
UCF101_ANNO_PATH = config["ucf101"]["anno"]
UCF101_WORKERS = config["ucf101"]["workers"]
UCF101_CACHE = config["ucf101"]["cache"]

# Dataloader Config Vars
DATALOADER_WORKERS = config["dataloader"]["workers"]

# Prediction Config Vars
PREDICTION_MODEL_CHECKPOINT = config["prediction"]["model"]
PREDICTION_MAX_EPOCHS = config["prediction"]["epochs"]
PREDICTION_OUTPUT_DIR = config["prediction"]["outdir"]

# Classification Config Vars
CLASSIFICATION_DATASET_PATH = config["classification"]["root"]
