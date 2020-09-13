from functools import partial

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from project.dataset.ucf101video import UCF101VideoDataModule
from project.train_video_prediction import SelfSupervisedVideoPredictionLitModel


class HyperparameterTuningWrapper(SelfSupervisedVideoPredictionLitModel):
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {
            "avg_train_loss": avg_loss,
        }


class TuneReportCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        tune.report(loss=trainer.callback_metrics["avg_train_loss"].item())


def train_ucf101_tune(config, datamodule, num_epochs=4):
    total_weight = config["l1_loss_wt"] + config["l2_loss_wt"] + config["ssim_loss_wt"]
    config["l1_loss_wt"] /= total_weight
    config["l2_loss_wt"] /= total_weight
    config["ssim_loss_wt"] /= total_weight

    print("Evaluating Params: ", config)

    model = HyperparameterTuningWrapper(datamodule=datamodule, **config)
    trainer = Trainer(
        checkpoint_callback=False,
        max_epochs=num_epochs,
        gpus=1,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        limit_train_batches=0.005,
        callbacks=[TuneReportCallback()],
    )
    trainer.fit(model)


def tune_ucf101_asha(
        config: dict,
        num_samples=50,
        num_epochs=4,
        gpus_per_trial=1,
        parameter_columns=None,
        metric_columns=None,
        folder_name="tune_ucf101_asha",
):
    ucf101_dm = UCF101VideoDataModule(batch_size=8)

    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=num_epochs, grace_period=1, reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=parameter_columns, metric_columns=metric_columns,
    )

    tune.run(
        partial(train_ucf101_tune, datamodule=ucf101_dm, num_epochs=num_epochs),
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=folder_name,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune Hyperparameters.")
    parser.add_argument(
        "integers",
        metavar="N",
        type=int,
        nargs="+",
        help="an integer for the accumulator",
    )
    parser.add_argument(
        "--params", type=str, choices=["loss", "lr"], help="which param to tune?"
    )

    args = parser.parse_args()

    if args.params == "loss":
        tune_ucf101_asha(
            config={
                "l1_loss_wt": tune.uniform(1, 100),
                "l2_loss_wt": tune.uniform(1, 100),
                "ssim_loss_wt": tune.uniform(1, 100),
            },
            parameter_columns=("l1_loss_wt", "l2_loss_wt", "ssim_loss_wt"),
            metric_columns=("loss", "training_iteration"),
            folder_name="tune_ucf101_losses"
        )
    elif args.params == "lr":
        tune_ucf101_asha(
            config={"lr": tune.loguniform(1e-4, 1e-2)},
            parameter_columns=("lr"),
            metric_columns=("loss", "training_iteration"),
            folder_name="tune_ucf101_lr"
        )
