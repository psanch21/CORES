from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

import lightning as L
import torch
import torch.optim.lr_scheduler as t_lr
import torch_geometric.data as pygd
from torch.optim import Optimizer

import cores.core.values.constants as cte
from cores.core.contracts.logger import Logger
from cores.impl.gnn.pyg.g_pool import GPool
from cores.impl.gnn.pyg.gnn_base import BaseGNN
from cores.impl.gnn.pyg.gnn_top_k import TopKGNN
from cores.impl.logger.dummy_logger import DummyLogger
from cores.impl.metrics import TaskMetricsTorch
from cores.impl.metrics.metric_tracker import MetricTrackerTorch
from cores.utils import PyGUtils, TorchUtils
from cores.utils.early_stopper import EarlyStopper
from cores.utils.plotter import Plotter


class TopKLightning(L.LightningModule):
    def __init__(
        self,
        graph_clf: BaseGNN,
        loss_fn: Callable,
        optimizer: cte.OptimizerType,
        optimizer_kwargs: dict[str, Any],
        lr_scheduler: cte.LRSchedulerType,
        lr_scheduler_kwargs: dict[str, Any],
        metric_objective: str,
        top_k_mode: str,
        top_k_kwargs: Dict[str, Any] = None,
        early_stopping_kwargs: Optional[Dict[str, Any]] = None,
        init_fn: Optional[Callable] = None,
    ):
        super(TopKLightning, self).__init__()

        if top_k_mode == "top_k":
            top_k_kwargs["in_channels"] = graph_clf.input_dim

            self.top_k = TopKGNN(
                **top_k_kwargs,
            )
        elif top_k_mode == "g_pool":
            top_k_kwargs["in_channels"] = graph_clf.input_dim

            self.top_k = GPool(
                **top_k_kwargs,
            )
        self.graph_clf = graph_clf

        self.loss_fn = loss_fn

        self.init_fn = init_fn

        self.optimizer = optimizer
        assert self.optimizer in cte.OptimizerType
        self.optim_kwargs = optimizer_kwargs

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.my_logger = DummyLogger()
        self.task_metrics = None

        self.metric_objective = metric_objective

        if early_stopping_kwargs is None:
            self.metric_clf = "valid/accuracy"
            self.early_stopper_clf = EarlyStopper(patience=50, min_delta=0.001, mode="max")
        else:
            clf_kwargs = early_stopping_kwargs
            self.metric_clf = clf_kwargs["metric"]
            del clf_kwargs["metric"]
            self.early_stopper_clf = EarlyStopper(**clf_kwargs)

        self.metric_tracker = MetricTrackerTorch()

        self.reset_parameters()

    def set_task_metrics(self, task_metrics: TaskMetricsTorch) -> None:
        self.task_metrics = task_metrics
        task_metrics_list = self.task_metrics.get_keys()
        self.task_metrics.remove()

        for metric in task_metrics_list:
            self.task_metrics.add(f"valid/{metric}")

    def set_my_logger(self, my_logger: Logger) -> None:
        self.my_logger = my_logger

    def reset_parameters(self):
        if self.init_fn is not None:
            self.graph_clf.apply(self.init_fn)
            self.top_k.apply(self.init_fn)

    def get_batch_norm(
        self,
        batch: pygd.Batch,
        x_noise: float = 0.0,
        edge_noise: float = 0.0,
    ):
        batch_norm = batch
        if edge_noise > 0.0:
            batch_norm = PyGUtils.add_edge_noise_batch(batch=batch_norm, p=edge_noise, sort=True)
        if x_noise > 0.0:
            batch_norm = PyGUtils.add_x_noise(batch=batch_norm, eps=x_noise)

        batch_norm = self.top_k(batch=batch_norm)

        return batch_norm

    @torch.no_grad()
    def predict(
        self,
        batch: pygd.Batch,
        x_noise: float = 0.0,
        edge_noise: float = 0.0,
        return_batch_norm: float = False,
        **kwargs,
    ):
        batch = batch.clone()
        stats_original = PyGUtils.compute_stats_batch(batch)

        batch_norm = self.get_batch_norm(
            batch,
            x_noise=x_noise,
            edge_noise=edge_noise,
        )

        stats_norm = PyGUtils.compute_stats_batch(batch_norm, num_samples=1)

        stats = {}
        for name, value in stats_original.items():
            value_2 = stats_norm[name]
            stats[f"{name}_ratio"] = value_2 / value

        logits = self.graph_clf(batch=batch_norm, *kwargs)
        target = batch_norm.y

        if return_batch_norm:
            return logits, target, stats, batch_norm
        else:
            return logits, target, stats

    def graph_clf_loss(self, batch: pygd.Batch, **kwargs):
        target = batch.y

        batch_norm = self.get_batch_norm(batch)
        logits = self.graph_clf(batch=batch_norm, *kwargs)

        loss = self.loss_fn(logits, target)

        return loss

    def on_fit_start(self) -> None:
        self.train()

        self.early_stopper_clf.reset()
        return super().on_fit_start()

    def on_train_epoch_start(self) -> None:
        self.metric_tracker.reset()
        self.task_metrics.reset()

        return super().on_train_epoch_start()

    @property
    def automatic_optimization(self):
        return False

    # process inside the training loop
    def training_step(self, train_batch: pygd.Batch, batch_idx: int):
        opt_clf = self.optimizers()

        tic = time.time()

        opt_clf.zero_grad()

        loss = self.graph_clf_loss(batch=train_batch)

        self.manual_backward(loss.mean())
        opt_clf.step()

        self.metric_tracker.update("training/loss", loss.item())

        self.metric_tracker.update("training/step_time", time.time() - tic)

        return

    def on_train_epoch_end(self) -> None:
        metrics_dict = self.metric_tracker.compute(regex="training.*")
        task_metrics_dict = self.task_metrics.compute(regex="training.*")
        metrics_dict.update(task_metrics_dict)

        metrics_dict["training/epoch"] = self.current_epoch

        opt = self.optimizers()
        if isinstance(opt, list):
            for i, o in enumerate(opt):
                metrics_dict[f"training/lr_{i}"] = o.optimizer.param_groups[0]["lr"]
        else:
            metrics_dict["training/lr"] = opt.optimizer.param_groups[0]["lr"]
        schedulers = self.lr_schedulers()

        self.do_scheduler_step(sch=schedulers, monitor=None, epoch_type="train")

        self.my_logger.track_data(metrics_dict)
        self.metric_tracker.reset()
        self.my_logger.increment_step()

        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.metric_tracker.reset(regex="valid")
        self.task_metrics.reset(regex="valid")
        return super().on_validation_epoch_start()

    def validation_step(self, batch: pygd.Batch, batch_idx: int):
        self.top_k.eval()
        self.graph_clf.eval()

        logits, target, stats = self.predict(batch)

        loss = self.loss_fn(logits, target)

        self.metric_tracker.update("valid/loss", loss.item())

        for name, value in stats.items():
            self.metric_tracker.update(f"valid/1__{name}", value)

        for key in self.task_metrics.get_keys(filter="valid.*"):
            self.task_metrics.update(key, logits=logits, target=target)

        return

    def on_validation_epoch_end(self) -> None:
        metrics_dict = self.metric_tracker.compute(regex="valid.*")
        task_metrics_dict = self.task_metrics.compute(regex="valid.*")
        metrics_dict.update(task_metrics_dict)

        # for k, v in metrics_dict.items():
        #     self.log(k, v.item())

        metrics_dict["valid/epoch"] = self.current_epoch

        self.my_logger.track_data(metrics_dict)

        sch = self.lr_schedulers()

        self.do_scheduler_step(
            sch=sch, monitor=metrics_dict[self.metric_objective], epoch_type="val"
        )

        metrics_dict_ = self.metric_tracker.compute()
        task_metrics_dict_ = self.metric_tracker.compute()
        metrics_dict_.update(task_metrics_dict_)
        metrics_dict.update(metrics_dict_)

        early_stopper_metric = metrics_dict[self.metric_clf]
        if self.early_stopper_clf.should_stop(early_stopper_metric):
            self.early_stopper_clf.reset()

            self.trainer.should_stop = True

        return super().on_validation_epoch_end()

    def logits_to_hard_pred(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1 or logits.shape[1] == 1:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=-1)

    def evaluate(
        self,
        loader: pygd.DataLoader,
        ckpt_file: Optional[str] = None,
        task_metrics: TaskMetricsTorch = None,
        plotter: Optional[Plotter] = None,
        images_folder: Optional[str] = None,
        device: str = "cpu",
    ):
        if ckpt_file is not None:
            ckpt_data = torch.load(ckpt_file)
            self.load_state_dict(ckpt_data["state_dict"])
        self.eval()

        task_metrics.reset()
        metric_tracker = MetricTrackerTorch()
        metric_tracker.reset()

        output_metrics = {}

        policy_dict = {
            "1": {"sample": True, "num_samples": 1},
        }

        if plotter is not None and images_folder is not None:
            os.makedirs(images_folder, exist_ok=True)

            for policy_key in policy_dict.keys():
                if policy_key == "1":
                    output_metrics[f"{policy_key}__plot"] = []

        plot_columns = ["data", "1__data_norm", "graph_id", "y", "1__y_pred"]
        plot_data = []

        self.to(device)

        for policy_key, policy_kwargs in policy_dict.items():
            task_metrics.reset()
            metric_tracker.reset()
            for batch_id, batch in enumerate(loader):
                batch = batch.to(device)
                logits, target, stats, batch_norm = self.predict(batch, return_batch_norm=True)

                loss = self.loss_fn(logits, target)
                metric_tracker.update("loss", loss.item())

                for stat_key, stat_value in stats.items():
                    metric_tracker.update(stat_key, stat_value)
                for metric_name in task_metrics.get_keys():
                    task_metrics.update(metric_name, logits=logits, target=target)

            task_metrics_dict = task_metrics.compute()
            for metric_name, metric_value in task_metrics_dict.items():
                output_metrics[f"{policy_key}__{metric_name}"] = metric_value
            metrics_dict = metric_tracker.compute()
            for metric_name, metric_value in metrics_dict.items():
                output_metrics[f"{policy_key}__{metric_name}"] = metric_value

        plot_dict = {
            "columns": plot_columns,
            "data": plot_data,
        }

        return output_metrics, plot_dict

    def configure_optimizers(self) -> Optimizer:
        opt_clf = {}

        # Set up Graph CLF
        params_clf = self.graph_clf.parameters()
        params_topk = self.top_k.parameters()
        params = list(params_clf) + list(params_topk)

        opt_clf["optimizer"] = TorchUtils.build_optimizer(
            name=self.optimizer, optim_kwargs=self.optim_kwargs, params=params
        )

        if self.lr_scheduler != cte.LRSchedulerType.NONE:
            opt_clf["lr_scheduler"] = TorchUtils.build_scheduler(
                name=self.lr_scheduler,
                scheduler_kwargs=self.lr_scheduler_kwargs,
                optimizer=opt_clf["optimizer"],
            )
            opt_clf["monitor"] = self.metric_objective

        return opt_clf

    def do_scheduler_step(self, sch: t_lr.LRScheduler, monitor: Optional[str], epoch_type: str):

        if epoch_type == "train":
            if isinstance(sch, list):
                for i, sch_i in enumerate(sch):
                    if not isinstance(sch_i, t_lr.ReduceLROnPlateau):
                        sch_i.step()
            elif sch is not None and not isinstance(sch, t_lr.ReduceLROnPlateau):
                sch.step()
        elif epoch_type == "val":
            if isinstance(sch, list):
                for i, sch_i in enumerate(sch):
                    if isinstance(sch_i, t_lr.ReduceLROnPlateau):
                        sch_i.step(monitor)
            elif sch is not None and isinstance(sch, t_lr.ReduceLROnPlateau):
                sch.step(monitor)
