import os
import random
import sys

import torch
from torch.nn import Upsample
from torch.utils.data.dataloader import default_collate

from project.config.user_config import PREDICTION_MODEL_CHECKPOINT


def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


def collate_bouncing_balls(batch):
    filtered_batch = []
    for sequence in batch:
        # for _ in range(4):
        startIdx = sequence.shape[0] // 2
        step = 3
        stopIdx = startIdx + step * 6
        # if stopIdx < sequence.shape[0]:
        new_sequence = sequence[startIdx:stopIdx:step, :, :]
        new_sequence = new_sequence.unsqueeze(dim=1)
        new_sequence = new_sequence.repeat(1, 3, 1, 1)
        filtered_batch.append(new_sequence)
    return torch.utils.data.dataloader.default_collate(filtered_batch)


double_resolution = Upsample(scale_factor=2, mode="bilinear", align_corners=True)


def load_model(clazz, **kwargs):
    if os.path.exists(PREDICTION_MODEL_CHECKPOINT):
        return clazz.load_from_checkpoint(PREDICTION_MODEL_CHECKPOINT, **kwargs)
    else:
        print("Error! Cannot load checkpoint at {}")
        sys.exit(-1)
