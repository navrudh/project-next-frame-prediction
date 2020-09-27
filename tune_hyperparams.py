from functools import partial

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from project.train_video_prediction import SelfSupervisedVideoPredictionLitModel


class HyperparameterTuningWrapper(SelfSupervisedVideoPredictionLitModel):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
        return [optimizer], [scheduler]

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {
            "avg_train_loss": avg_loss,
        }

    def val_dataloader(self):
        """"""

    def validation_step(self, batch, batch_nb):
        """"""

    def validation_epoch_end(self, outputs):
        """"""


class TuneReportCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        tune.report(loss=trainer.callback_metrics["avg_train_loss"])


def train_ucf101_tune(config, num_epochs):
    if "l1_loss_wt" in config:
        total_weight = (
            config["l1_loss_wt"] + config["l2_loss_wt"] + config["ssim_loss_wt"]
        )
        config["l1_loss_wt"] /= total_weight
        config["l2_loss_wt"] /= total_weight
        config["ssim_loss_wt"] /= total_weight

    print("Evaluating Params: ", config)

    model = HyperparameterTuningWrapper(batch_size=4, **config)
    trainer = Trainer(
        checkpoint_callback=False,
        max_epochs=num_epochs,
        gpus=1,
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        val_percent_check=0.0,
        limit_train_batches=0.005,
        callbacks=[TuneReportCallback()],
    )
    trainer.fit(model)


def tune_ucf101_asha(
    config: dict,
    num_samples=50,
    num_epochs=12,
    gpus_per_trial=1,
    parameter_columns=None,
    metric_columns=None,
    folder_name="tune_ucf101_asha",
):
    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=num_epochs, grace_period=1, reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=parameter_columns, metric_columns=metric_columns,
    )

    tune.run(
        partial(train_ucf101_tune, num_epochs=num_epochs),
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
        "--params",
        type=str,
        choices=["loss", "lr"],
        help="which param to tune?",
        required=True,
    )

    args = parser.parse_args()

    if args.params == "loss":
        tune_ucf101_asha(
            config={
                "l1_loss_wt": tune.uniform(1, 100),
                "l2_loss_wt": tune.uniform(1, 100),
                "ssim_loss_wt": tune.uniform(1, 100),
            },
            parameter_columns=["l1_loss_wt", "l2_loss_wt", "ssim_loss_wt"],
            metric_columns=["loss", "training_iteration"],
            folder_name="tune_ucf101_losses",
            num_samples=30,
            num_epochs=12,
        )
    elif args.params == "lr":
        tune_ucf101_asha(
            config={"lr": tune.loguniform(1e-4, 1e-2)},
            parameter_columns=["lr"],
            metric_columns=["loss", "training_iteration"],
            folder_name="tune_ucf101_lr",
            num_samples=20,
        )
