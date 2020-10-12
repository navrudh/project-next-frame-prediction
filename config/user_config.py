import copy
import getpass
import json
import os
import pathlib

username = getpass.getuser()
current_py_file_path = os.path.dirname(os.path.abspath(__file__))
settings_dir = os.path.abspath(
    os.path.join(current_py_file_path, "../resources/config")
)
config = json.load(open(f"{settings_dir}/user-{username}.json"))

## Dataset Config Vars
# UCF101
UCF101_PATH = config["ucf101"]["path"]
UCF101_ROOT_PATH = UCF101_PATH + "/UCF101/UCF-101"
if len(config["ucf101"].get("custom-anno-split", "")) > 0:
    UCF101_ANNO_SUFFIX = "-" + config["ucf101"]["custom-anno-split"]
else:
    UCF101_ANNO_SUFFIX = ""
UCF101_ANNO_PATH = (
    UCF101_PATH
    + "/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist"
    + UCF101_ANNO_SUFFIX
)

UCF101_WORKERS = config["ucf101"]["workers"]

# Bouncing Balls
BB_SIZE = config["bouncing-balls"]["size"]
BB_NBALLS = config["bouncing-balls"]["n-balls"]
BB_TIMESTEPS = config["bouncing-balls"]["timesteps"]

# Dataloader Config Vars
DATALOADER_WORKERS = config["dataloader"]["workers"]

# Work Dir
WORK_DIR = config["workdir"]

# Save Config
CONFIG_FILE = WORK_DIR + "/config.json"
SAVE_CFG_KEY_DATASET = "dataset-used"

# Prediction Config Vars
PREDICTION_MODEL_CHECKPOINT = WORK_DIR + "/model.ckpt"
PREDICTION_OUTPUT_DIR = WORK_DIR + "/generated"
PREDICTION_BATCH_SIZE = config["prediction"]["batch_size"]
PREDICTION_LR = config["prediction"]["learning_rate"]
PREDICTION_DECAY = config["prediction"]["weight_decay"]
PREDICTION_PATIENCE = config["prediction"]["sched_patience"]
PREDICTION_SCHED_FACTOR = config["prediction"]["sched_factor"]
PREDICTION_TRAINER_KWARGS = config["prediction"]["trainer_args"]

# Classification Config Vars
CLASSIFICATION_DATASET_PATH = WORK_DIR + "/classification/tensors"


def save_config(additional_config=None):
    _config = copy.deepcopy(config)
    file = pathlib.Path(CONFIG_FILE)
    if file.exists():
        print("Config exists. Should the workdir be cleaned?")
    if additional_config:
        _config = {**_config, **additional_config}
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as fp:
        json.dump(_config, fp, sort_keys=True, indent=4)


def load_saved_config():
    file = pathlib.Path(CONFIG_FILE)
    if not file.exists():
        raise Exception(f"Missing config: {file}. Has training been run?")
    return json.load(open(file))
