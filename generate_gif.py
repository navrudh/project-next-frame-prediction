import uuid

import torch
import torchvision
from pytorch_lightning import Trainer

from project.config.user_config import PREDICTION_OUTPUT_DIR
from project.dataset.ucf101video import invert_transforms, UCF101VideoDataModule
from project.train_video_prediction import SelfSupervisedVideoPredictionLitModel
from project.utils.image import generate_gif
from project.utils.train import double_resolution, load_model


class GifGenerator(SelfSupervisedVideoPredictionLitModel):
    def test_step(self, batch, batch_nb):
        curr, next, pred, loss = self.predict_frame(batch, batch_nb)
        pred = double_resolution(pred)

        # Undo transforms / fix colors
        curr = invert_transforms(curr)
        pred = invert_transforms(pred)

        # Group Sequences
        curr = curr.view(-1, 5, 3, self.image_dim, self.image_dim)
        pred = pred.view(-1, 3, 3, self.image_dim, self.image_dim)

        # Collect Frames
        sequence = torch.zeros(
            (curr.shape[0], 6, 3, self.image_dim, self.image_dim), device=self.device
        )

        sequence[:, :3, :, :, :] = curr[:, :3, :, :, :]
        sequence[:, 3, :, :, :] = pred[:, -3, :, :, :]
        sequence[:, 4, :, :, :] = pred[:, -2, :, :, :]
        sequence[:, 5, :, :, :] = pred[:, -1, :, :, :]

        dim = sequence.shape[0]
        batch_tensors = torch.unbind(sequence)
        for i in range(dim):
            file_name = uuid.uuid1()
            for seq_no, image_tensor in enumerate(batch_tensors[i]):
                torchvision.utils.save_image(
                    image_tensor,
                    fp=f"{PREDICTION_OUTPUT_DIR}/{file_name}-{seq_no}.jpg",
                    normalize=True,
                )
            generate_gif(
                PREDICTION_OUTPUT_DIR,
                file_glob=f"{file_name}*.jpg",
                gif_name=f"{PREDICTION_OUTPUT_DIR}/{file_name}.gif",
            )


if __name__ == "__main__":
    ucf101_dm = UCF101VideoDataModule(batch_size=8)
    lit_model = load_model(GifGenerator, batch_size=8)
    lit_model.eval()
    ucf101_dm.setup("test")

    trainer = Trainer(logger=False, gpus=1, limit_test_batches=0.025)
    trainer.test(lit_model, test_dataloaders=ucf101_dm.test_dataloader())

    # trainer.test(
    #     ckpt_path=PREDICTION_MODEL_CHECKPOINT,
    #     test_dataloaders=ucf101_dm.test_dataloader(),
    # )
