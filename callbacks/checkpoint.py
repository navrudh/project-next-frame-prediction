import pytorch_lightning as pl


class SaveCheckpointAtEpochEnd(pl.Callback):
    def __init__(self, filepath):
        self.filepath = filepath

    def on_epoch_end(self, trainer, pl_module):
        trainer.checkpoint_callback._save_model(self.filepath, trainer, pl_module)
