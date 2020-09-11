import torch
from torch.nn import Upsample
from torch.utils.data.dataloader import default_collate


def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


double_resolution = Upsample(scale_factor=2, mode="bilinear")
