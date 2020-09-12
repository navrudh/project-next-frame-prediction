from functools import partial

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from project.dataset.ucf101video import UCF101VideoDataModule
from project.train_video_prediction import SelfSupervisedVideoPredictionLitModel


class TuneReportCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        tune.report(loss=trainer.callback_metrics["loss"].item())


def train_mnist_tune(config, datamodule, num_epochs=4):
    total_weight = config["l1_loss_wt"] + config["l2_loss_wt"] + config["ssim_loss_wt"]
    config["l1_loss_wt"] /= total_weight
    config["l2_loss_wt"] /= total_weight
    config["ssim_loss_wt"] /= total_weight

    print("Evaluating Params: ", config)

    model = SelfSupervisedVideoPredictionLitModel(datamodule=datamodule, **config)
    trainer = Trainer(
        checkpoint_callback=False,
        max_epochs=num_epochs,
        gpus=1,
        logger=TensorBoardLogger("lightning_logs", name="tune"),
        # progress_bar_refresh_rate=0,
        limit_train_batches=0.001,
        callbacks=[TuneReportCallback()],
    )
    trainer.fit(model)


def tune_mnist_asha(num_samples=50, num_epochs=4, gpus_per_trial=1):
    ucf101_dm = UCF101VideoDataModule(batch_size=8)

    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "l1_loss_wt": tune.uniform(1, 100),
        "l2_loss_wt": tune.uniform(1, 100),
        "ssim_loss_wt": tune.uniform(1, 100),
    }

    scheduler = ASHAScheduler(
        metric="loss", mode="min", max_t=num_epochs, grace_period=1, reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=["l1_loss_wt", "l2_loss_wt", "ssim_loss_wt", "lr"],
        metric_columns=["loss", "training_iteration"],
    )

    tune.run(
        partial(train_mnist_tune, datamodule=ucf101_dm, num_epochs=num_epochs),
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha",
    )


if __name__ == "__main__":
    tune_mnist_asha()
