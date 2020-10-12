import os
import sys

import torch
from torch.nn import Upsample
from torch.utils.data.dataloader import default_collate

from config.user_config import PREDICTION_MODEL_CHECKPOINT


def collate_ucf101(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


def collate_bouncing_balls(batch):
    filtered_batch = []
    for sequence in batch:
        filtered_batch.append((sequence, -1))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


double_resolution = Upsample(scale_factor=2, mode="bilinear", align_corners=True)


def load_model(clazz, **kwargs):
    if os.path.exists(PREDICTION_MODEL_CHECKPOINT):
        return clazz.load_from_checkpoint(PREDICTION_MODEL_CHECKPOINT, **kwargs)
    else:
        print("Error! Cannot load checkpoint at {}")
        sys.exit(-1)
