import time
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, Timer

class TimeLogger(Callback):
    def __init__(self, timer: Timer):
        super().__init__()
        self.timer = timer

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.train_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pl_module.log(
            "train/epoch_time",
            time.perf_counter() - self.train_start,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        trainer.logger.log_metrics(
            {
                "train/total_time": self.timer.time_elapsed("train"),
            },
            step=trainer.global_step,
        )