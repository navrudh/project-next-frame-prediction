import getpass
import json
import uuid

import torch
import torchvision
from torch.nn import Upsample

from project.train_video_prediction import (
    SelfSupervisedVideoPredictionLitModel,
    load_or_train_model,
)
from project.utils.info import print_device, seed

print_device()
seed(42)

username = getpass.getuser()
config = json.load(open(f"{username}.json"))
GIF_OUT = config["prediction"]["outdir"]
CHECKPOINT_PATH = config["prediction"]["model"]

increase_res = Upsample(scale_factor=2, mode="bilinear")


class GifGenerator(SelfSupervisedVideoPredictionLitModel):
    def test_step(self, batch, batch_nb):
        # Save Hidden Layers under batch numbers
        x, y = batch
        batch_size = x.shape[0]

        # Predict Frames
        curr = torch.zeros(
            (x.shape[0], 5, 3, self.image_dim, self.image_dim), device=self.device
        )
        curr[:, :3, :, :, :] = x[:, :3, :, :, :]
        curr = curr.reshape(-1, 3, self.image_dim, self.image_dim).contiguous()
        pred = self.model.forward(curr)
        pred = increase_res(pred)
        pred = pred.reshape(batch_size, -1, 3, self.image_dim, self.image_dim)

        # Collect Frames
        sequence = torch.zeros(
            (x.shape[0], 6, 3, self.image_dim, self.image_dim), device=self.device
        )
        sequence[:, :3, :, :, :] = x[:, :3, :, :, :]
        sequence[:, 3, :, :, :] = pred[:, 0, :, :, :]
        sequence[:, 4, :, :, :] = pred[:, 1, :, :, :]
        sequence[:, 5, :, :, :] = pred[:, 2, :, :, :]

        dim = sequence.shape[0]
        batch_tensors = torch.unbind(sequence)
        for i in range(dim):
            file_name = uuid.uuid1()
            for seq_no, image_tensor in enumerate(batch_tensors[i]):
                torchvision.utils.save_image(
                    image_tensor, fp=f"{GIF_OUT}/{file_name}-{seq_no}.jpg"
                )


lit_model = GifGenerator(hidden_dims=[64, 64, 128, 256], batch_size=8)

lit_model, trainer = load_or_train_model(
    lit_model, tensorboard_graph_name=None, gif_mode=True, save=False
)

trainer.test(ckpt_path=CHECKPOINT_PATH, test_dataloaders=lit_model.test_dataloader())
