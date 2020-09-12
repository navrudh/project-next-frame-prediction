import os

import numpy as np
import torch
from tqdm import tqdm

from project.config.user_config import UCF101_CACHE
from project.dataset.ucf101video import UCF101VideoDataModule


def save_batch(step_name, batch, batch_nb, batch_size, class_to_idx):
    x, y = batch
    dim = x.shape[0]
    video_frames_tensors = torch.unbind(x)
    for i in range(dim):
        os.makedirs(
            f"{UCF101_CACHE}/{step_name}/{class_to_idx[y[i].item()]}", exist_ok=True
        )
        np.save(
            f"{UCF101_CACHE}/{step_name}/{class_to_idx[y[i].item()]}/{batch_size * batch_nb + i}.pt",
            video_frames_tensors[i],
        )


ucf101_dm = UCF101VideoDataModule(batch_size=8)
ucf101_dm.setup("test")
# for batch_nb, batch in tqdm(enumerate(ucf101_dm.train_dataloader())):
#     save_batch('train_dataset', batch, batch_nb, ucf101_dm.batch_size, ucf101_dm.class_to_idx)
for batch_nb, batch in enumerate(tqdm(ucf101_dm.test_dataloader())):
    save_batch(
        "test_dataset", batch, batch_nb, ucf101_dm.batch_size, ucf101_dm.class_to_idx
    )
