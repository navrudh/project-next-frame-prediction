import uuid

import torch
import torchvision

from project.config.user_config import (
    PREDICTION_MODEL_CHECKPOINT,
    PREDICTION_OUTPUT_DIR,
)
from project.dataset.ucf101video import UCF101VideoDataModule
from project.train_video_prediction import (
    SelfSupervisedVideoPredictionLitModel,
    load_or_train_model,
)
from project.utils.image import generate_gif
from project.utils.info import print_device, seed
from project.utils.train import double_resolution

print_device()
seed(42)


class GifGenerator(SelfSupervisedVideoPredictionLitModel):
    def test_step(self, batch, batch_nb):
        # Save Hidden Layers under batch numbers
        x, y = batch
        batch_size = x.shape[0]

        # Predict Frames
        curr = torch.zeros(
            (x.shape[0], 5, 3, self.image_dim, self.image_dim), device=self.device
        )
        curr[:, :5, :, :, :] = x[:, :5, :, :, :]
        curr = curr.reshape(-1, 3, self.image_dim, self.image_dim).contiguous()
        pred = self.model.forward(curr)
        pred = double_resolution(pred)
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
                    image_tensor, fp=f"{PREDICTION_OUTPUT_DIR}/{file_name}-{seq_no}.jpg"
                )
            generate_gif(
                PREDICTION_OUTPUT_DIR,
                file_glob=f"{file_name}*.jpg",
                gif_name=f"{PREDICTION_OUTPUT_DIR}/{file_name}.gif",
            )


if __name__ == "__main__":
    ucf101_dm = UCF101VideoDataModule(batch_size=8)
    lit_model = GifGenerator(ucf101_dm)

    lit_model, trainer = load_or_train_model(
        lit_model, tensorboard_graph_name=None, gif_mode=True, save=False
    )

    trainer.test(
        ckpt_path=PREDICTION_MODEL_CHECKPOINT,
        test_dataloaders=lit_model.test_dataloader(),
    )
