import os

import torch
import torchvision
from pytorch_lightning import Trainer

from config.user_config import (
    PREDICTION_OUTPUT_DIR,
    load_saved_config,
    SAVE_CFG_KEY_DATASET,
    PREDICTION_BATCH_SIZE,
)
from dataset.bouncing_balls_video import BouncingBallsVideoDataModule
from dataset.ucf101video import UCF101VideoDataModule
from train_video_prediction_bouncing_balls import BouncingBallsVideoPredictionLitModel
from train_video_prediction_ucf101 import UCF101VideoPredictionLitModel
from utils.image import generate_gif
from utils.train import load_model

OUTPUT_DIR = PREDICTION_OUTPUT_DIR


def get_inherited_gif_generator_class(BaseLitModel):
    class GifGenerator(BaseLitModel):
        def test_step(self, batch, batch_nb):
            inp, pred, loss, hidden = self.predict_frame(batch, batch_nb)

            # pred = double_resolution(pred)

            # Undo transforms / fix colors
            # curr = invert_transforms(curr)
            # pred = invert_transforms(pred)

            # Group Sequences
            inp = inp.view(-1, 6, *inp.shape[-3:])
            pred = pred.view(-1, 6, *pred.shape[-3:])

            # Collect Frames
            # sequence = torch.zeros(
            #     inp.shape, device=self.device
            # )
            #
            # sequence[:, :3, :, :, :] = inp[:, :3, :, :, :]
            # sequence[:, 3, :, :, :] = pred[:, -3, :, :, :]
            # sequence[:, 4, :, :, :] = pred[:, -2, :, :, :]
            # sequence[:, 5, :, :, :] = pred[:, -1, :, :, :]

            dim = inp.shape[0]
            unbinded_preds = torch.unbind(pred)
            unbinded_inps = torch.unbind(inp)
            for i in range(dim):
                file_name = f"pred-{batch_nb:03}-{i}"
                for seq_no, image_tensor in enumerate(
                    zip(unbinded_inps[i], unbinded_preds[i])
                ):
                    torchvision.utils.save_image(
                        list(image_tensor),
                        fp=f"{OUTPUT_DIR}/{file_name}-{seq_no}.jpg",
                        normalize=True,
                    )
                generate_gif(
                    OUTPUT_DIR,
                    file_glob=f"{file_name}*.jpg",
                    gif_name=f"{OUTPUT_DIR}/{file_name}.gif",
                )

    return GifGenerator


if __name__ == "__main__":
    saved_config = load_saved_config()
    dataset = saved_config[SAVE_CFG_KEY_DATASET]
    print("Dataset:", dataset)

    if dataset == "ucf101":
        datamodule = UCF101VideoDataModule(batch_size=PREDICTION_BATCH_SIZE)
        lit_model = load_model(
            get_inherited_gif_generator_class(UCF101VideoPredictionLitModel),
            batch_size=PREDICTION_BATCH_SIZE,
        )

    elif dataset == "bouncing-balls":
        datamodule = BouncingBallsVideoDataModule(batch_size=PREDICTION_BATCH_SIZE)
        lit_model = load_model(
            get_inherited_gif_generator_class(BouncingBallsVideoPredictionLitModel),
            batch_size=PREDICTION_BATCH_SIZE,
        )

    else:
        raise Exception("Dataset not supported: " + dataset)

    lit_model.eval()
    datamodule.setup()
    trainer = Trainer(logger=False, gpus=1, limit_test_batches=0.025)

    for split_name, loader in (
        ("train", datamodule.train_dataloader()),
        ("test", datamodule.val_dataloader()),
    ):
        print("Loader:", split_name)
        OUTPUT_DIR = PREDICTION_OUTPUT_DIR + "/" + split_name
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        trainer.test(lit_model, test_dataloaders=loader)
