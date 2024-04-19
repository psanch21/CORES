from __future__ import annotations

import time
from typing import Any, Callable, Optional

import lightning as L
import torch
import torch_geometric.data as pygd
from torch.optim import Optimizer

import cores.core.values.constants as cte
from cores.core.contracts.logger import Logger
from cores.impl.gnn.pyg.gnn_base import BaseGNN
from cores.impl.logger.dummy_logger import DummyLogger
from cores.impl.metrics import TaskMetricsTorch
from cores.impl.metrics.metric_tracker import MetricTrackerTorch
from cores.utils import TorchUtils


class GNNLightning(L.LightningModule):
    def __init__(
        self,
        model: BaseGNN,
        optimizer: cte.OptimizerType,
        optimizer_kwargs: dict[str, Any],
        lr_scheduler: cte.LRSchedulerType,
        lr_scheduler_kwargs: dict[str, Any],
        loss_fn: Callable,
        init_fn: Optional[Callable] = None,
    ):
        super(GNNLightning, self).__init__()

        self.model = model

        self.init_fn = init_fn

        self.optimizer_name = optimizer
        assert self.optimizer_name in cte.OptimizerType
        self.optimizer_kwargs = optimizer_kwargs

        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.loss_fn = loss_fn

        self.my_logger = DummyLogger()
        self.task_metrics = None

        self.metric_tracker = MetricTrackerTorch()
        self.metric_tracker.add("training/loss")
        self.metric_tracker.add("training/time_step")
        self.metric_tracker.add("valid/loss")
        self.metric_tracker.add("valid/time_step")

    def set_task_metrics(self, task_metrics: TaskMetricsTorch) -> None:
        self.task_metrics = task_metrics
        task_metrics_list = self.task_metrics.get_keys()
        self.task_metrics.remove()

        for metric in task_metrics_list:
            self.task_metrics.add(f"training/{metric}")
            self.task_metrics.add(f"valid/{metric}")

    def set_my_logger(self, my_logger: Logger) -> None:
        self.my_logger = my_logger

    def reset_parameters(self) -> None:
        if self.init_fn is not None:
            self.model.apply(self.init_fn)
        return

    def forward(self, batch: pygd.Batch) -> torch.Tensor:
        logits = self.model(batch)

        return logits

    def configure_optimizers(self) -> Optimizer:
        optimizer = TorchUtils.build_optimizer(
            name=self.optimizer_name,
            optim_kwargs=self.optimizer_kwargs,
            params=self.model.parameters(),
        )

        # Define your LR scheduler

        scheduler = TorchUtils.build_scheduler(
            name=self.lr_scheduler_name,
            scheduler_kwargs=self.lr_scheduler_kwargs,
            optimizer=optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",  # Monitor the training loss for LR scheduling
            },
        }

    def on_fit_start(self) -> None:
        self.reset_parameters()
        return super().on_fit_start()

    def on_train_epoch_start(self) -> None:
        self.metric_tracker.reset()
        self.task_metrics.reset()
        return super().on_train_epoch_start()

    def training_step(self, batch: pygd.Batch, batch_idx: int) -> torch.Tensor:
        tic = time.time()
        logits = self.model(batch)
        loss = self.loss_fn(logits, batch.y)
        training_step_time = time.time() - tic

        self.metric_tracker.update("training/loss", loss.item())
        self.metric_tracker.update("training/time_step", training_step_time)

        # logits = logits.detach()
        # for metric in self.task_metrics.get_keys("training"):
        #     self.task_metrics.update(metric, logits, batch.y)

        return loss

    def on_train_epoch_end(self) -> None:
        metrics_dict = self.metric_tracker.compute()
        task_metrics_dict = self.task_metrics.compute()
        metrics_dict.update(task_metrics_dict)
        metrics_dict["training/epoch"] = self.current_epoch
        lr = self.lr_schedulers().get_last_lr()[0]
        metrics_dict["training/lr"] = lr
        self.my_logger.track_data(metrics_dict)
        self.metric_tracker.reset()
        self.my_logger.increment_step()
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.metric_tracker.reset(regex="valid")
        return super().on_validation_epoch_start()

    def validation_step(self, batch: pygd.Batch, batch_idx: int):
        tic = time.time()

        logits = self.model(batch)

        loss = self.loss_fn(logits, batch.y)

        valid_step_time = time.time() - tic

        self.metric_tracker.update("valid/loss", loss.item())
        self.metric_tracker.update("valid/time_step", valid_step_time)

        for key in self.task_metrics.get_keys(filter="valid.*full"):
            self.task_metrics.update(key=key, logits=logits, target=batch.y)

        return loss

    def on_validation_epoch_end(self) -> None:
        metrics_dict = self.metric_tracker.compute()
        task_metrics_dict = self.task_metrics.compute()
        metrics_dict.update(task_metrics_dict)
        for k, v in metrics_dict.items():
            self.log(k, v.item())
        metrics_dict["valid/epoch"] = self.current_epoch

        self.my_logger.track_data(metrics_dict)

        return super().on_validation_epoch_end()
