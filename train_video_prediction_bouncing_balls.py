from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from config.user_config import (
    BB_SIZE,
    BB_TIMESTEPS,
    BB_NBALLS,
    save_config,
    SAVE_CFG_KEY_DATASET,
    PREDICTION_BATCH_SIZE,
    DATALOADER_WORKERS,
    PREDICTION_LR,
    PREDICTION_DECAY,
    PREDICTION_PATIENCE,
    PREDICTION_SCHED_FACTOR,
)
from config.user_config import WORK_DIR
from dataset.bouncing_balls import BouncingBalls
from train_video_prediction_ucf101 import (
    UCF101VideoPredictionLitModel,
    load_or_train_model,
)
from transforms.video import augment_bouncing_balls_video_frames
from utils.train import collate_bouncing_balls


class BouncingBallsVideoPredictionLitModel(UCF101VideoPredictionLitModel):
    def dataset_init(self, stage):
        if stage == "train" and not self.train_dataset:
            self.train_dataset = BouncingBalls(
                size=BB_SIZE,
                timesteps=BB_TIMESTEPS,
                n_balls=BB_NBALLS,
                mode="train",
                train_size=500,
                transform=transforms.Lambda(augment_bouncing_balls_video_frames),
            )

        if stage == "val" and not self.val_dataset:
            self.val_dataset = BouncingBalls(
                size=BB_SIZE,
                timesteps=BB_TIMESTEPS,
                n_balls=BB_NBALLS,
                mode="val",
                train_size=500,
                # output_frames=6,
            )

    def train_dataloader(self):
        print("Train Dataloader Called")
        self.dataset_init("train")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_bouncing_balls,
        )

    def val_dataloader(self):
        print("Val Dataloader Called")
        self.dataset_init("val")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=DATALOADER_WORKERS,
            pin_memory=True,
            collate_fn=collate_bouncing_balls,
        )


if __name__ == "__main__":
    additional_config = {SAVE_CFG_KEY_DATASET: "bouncing-balls"}
    save_config(additional_config)
    lit_model = BouncingBallsVideoPredictionLitModel(
        batch_size=PREDICTION_BATCH_SIZE,
        lr=PREDICTION_LR,
        wt_decay=PREDICTION_DECAY,
        sched_patience=PREDICTION_PATIENCE,
        sched_factor=PREDICTION_SCHED_FACTOR,
    )
    lit_model, trainer = load_or_train_model(
        lit_model,
        tensorboard_graph_name=WORK_DIR.split("/")[-1],
        # resume=False,
        # save=False,
        # validation=False,
    )
    print("Completed Video Prediction Training")
