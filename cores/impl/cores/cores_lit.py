from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

import lightning as L
import numpy as np
import torch
import torch.optim.lr_scheduler as t_lr
import torch_geometric.data as pygd
from torch.optim import Optimizer

import cores.core.values.constants as cte
from cores.core.contracts.logger import Logger
from cores.impl.gnn.pyg.gnn_base import BaseGNN
from cores.impl.logger.dummy_logger import DummyLogger
from cores.impl.metrics import TaskMetricsTorch
from cores.impl.metrics.metric_tracker import MetricTrackerTorch
from cores.impl.rl.graph.envs.env_base import GraphEnvBase
from cores.impl.rl.graph.ppo import GraphPPO
from cores.utils import PyGUtils, TorchUtils
from cores.utils.early_stopper import EarlyStopper
from cores.utils.plotter import Plotter


class CORESLightning(L.LightningModule):
    def __init__(
        self,
        env: GraphEnvBase,
        graph_clf: BaseGNN,
        ppo: GraphPPO,
        loss_fn: Callable,
        optimizer_clf: cte.OptimizerType,
        optimizer_kwargs_clf: dict[str, Any],
        lr_scheduler_clf: cte.LRSchedulerType,
        lr_scheduler_kwargs_clf: dict[str, Any],
        optimizer_rl: cte.OptimizerType,
        optimizer_kwargs_rl: dict[str, Any],
        lr_scheduler_rl: cte.LRSchedulerType,
        lr_scheduler_kwargs_rl: dict[str, Any],
        env_steps: int = 128,
        ppo_steps: int = 1,
        early_stopping_kwargs: Optional[Dict[str, Any]] = None,
        metric_objective: Dict[str, Any] = None,
        gnn_mode: bool = False,
        init_fn: Optional[Callable] = None,
    ):
        super(CORESLightning, self).__init__()

        self.env = env
        self.graph_clf = graph_clf
        self.ppo = ppo

        self.ppo_steps = ppo_steps

        self.n_steps = env_steps

        self.loss_fn = loss_fn

        self.init_fn = init_fn

        self.optimizer_clf = optimizer_clf
        assert self.optimizer_clf in cte.OptimizerType
        self.optim_kwargs_clf = optimizer_kwargs_clf

        self.lr_scheduler_clf = lr_scheduler_clf
        self.lr_scheduler_kwargs_clf = lr_scheduler_kwargs_clf

        self.optimizer_rl = optimizer_rl
        assert self.optimizer_rl in cte.OptimizerType
        self.optim_kwargs_rl = optimizer_kwargs_rl

        self.lr_scheduler_rl = lr_scheduler_rl
        self.lr_scheduler_kwargs_rl = lr_scheduler_kwargs_rl

        self.gnn_mode = gnn_mode

        self.bilevel_iteration_count = 0

        if metric_objective is None:
            metric_objective = {}
            if self.gnn_mode:
                metric_objective["name"] = "valid/full__accuracy"
                metric_objective["mode"] = "max"
            else:
                metric_objective["name"] = "valid/1__accuracy"
                metric_objective["mode"] = "max"

        self.my_logger = DummyLogger()
        self.task_metrics = None

        self.num_samples_list = [1]

        self.metric_objective = metric_objective["name"]

        if early_stopping_kwargs is None:
            self.metric_ppo = "training/ppo_rewards"
            self.early_stopper_rl = EarlyStopper(patience=5, min_delta=0.001, mode="max")

            self.metric_clf = "valid/full__accuracy"
            self.early_stopper_clf = EarlyStopper(patience=5, min_delta=0.001, mode="max")
        else:
            ppo_kwargs = early_stopping_kwargs["ppo_kwargs"]

            metric_ppo_name, metric_ppo_mode = ppo_kwargs["metric"].split("___")
            self.metric_ppo = metric_ppo_name
            ppo_kwargs["mode"] = metric_ppo_mode
            del ppo_kwargs["metric"]
            self.early_stopper_rl = EarlyStopper(**ppo_kwargs)

            clf_kwargs = early_stopping_kwargs["clf_kwargs"]
            self.metric_clf = clf_kwargs["metric"]
            del clf_kwargs["metric"]
            self.early_stopper_clf = EarlyStopper(**clf_kwargs)

        self.metric_tracker = MetricTrackerTorch()

        self.reset_parameters()

        self.train_ppo = False

        self.use_sparse_graph = False

    def set_task_metrics(self, task_metrics: TaskMetricsTorch) -> None:
        self.task_metrics = task_metrics
        task_metrics_list = self.task_metrics.get_keys()
        self.task_metrics.remove()

        for metric in task_metrics_list:
            for extra in ["full", *self.num_samples_list]:
                self.task_metrics.add(f"valid/{extra}__{metric}")

    def set_my_logger(self, my_logger: Logger) -> None:
        self.my_logger = my_logger

    def reset_parameters(self):
        if self.init_fn is not None:
            self.graph_clf.apply(self.init_fn)
            self.ppo.apply(self.init_fn)

    def get_batch_norm(
        self,
        batch: pygd.Batch,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        x_noise: float = 0.0,
        edge_noise: float = 0.0,
    ):
        batch_norm = batch.clone()
        if edge_noise > 0.0:
            batch_norm = PyGUtils.add_edge_noise_batch(batch=batch_norm, p=edge_noise, sort=True)
        if x_noise > 0.0:
            batch_norm = PyGUtils.add_x_noise(batch=batch_norm, eps=x_noise)
        if policy_kwargs is not None:

            batch_norm = self.ppo.run_episode(
                batch=batch_norm,
                env=self.env,
                sample=policy_kwargs["sample"],
                num_samples=policy_kwargs["num_samples"],
            )

        return batch_norm

    @torch.no_grad()
    def predict(
        self,
        batch: pygd.Batch,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        x_noise: float = 0.0,
        edge_noise: float = 0.0,
        return_batch_norm: float = False,
        **kwargs,
    ):
        batch = batch.clone()
        stats_original = PyGUtils.compute_stats_batch(batch)

        batch_norm = self.get_batch_norm(
            batch,
            policy_kwargs=policy_kwargs,
            x_noise=x_noise,
            edge_noise=edge_noise,
        )

        num_samples = 1
        if policy_kwargs is not None:
            num_samples = policy_kwargs["num_samples"]

        stats_norm = PyGUtils.compute_stats_batch(batch_norm, num_samples=num_samples)

        stats = {}
        for name, value in stats_original.items():
            value_2 = stats_norm[name]
            stats[f"{name}_ratio"] = value_2 / value

        logits = self.graph_clf(batch=batch_norm.to(self.device), *kwargs)
        target = batch_norm.y

        if return_batch_norm:
            return logits, target, stats, batch_norm
        else:
            return logits, target, stats

    def policy_loss_dict(self, shuffle: bool = False) -> Dict[str, torch.Tensor]:
        self.ppo.train()
        self.graph_clf.eval()
        assert self.ppo.training
        loss_dict_tmp = self.ppo(shuffle=shuffle)
        loss_dict = {}
        for key, value in loss_dict_tmp.items():
            loss_dict[f"ppo_{key}"] = value
        return loss_dict

    def graph_clf_loss(
        self, batch: pygd.Batch, policy_kwargs: Optional[Dict[str, Any]] = None, **kwargs
    ):
        target = batch.y

        batch_norm = self.get_batch_norm(batch, policy_kwargs=policy_kwargs)
        logits = self.graph_clf(batch=batch_norm.detach().clone(), *kwargs)

        loss = self.loss_fn(logits, target)

        return loss

    def on_fit_start(self) -> None:
        self.eval()
        if not self.gnn_mode:
            self.env.reward_fn.fit_conformal_from_loader(batch_norm_fn=self.get_batch_norm)
        self.train()

        self.early_stopper_clf.reset()
        self.early_stopper_rl.reset()
        self.train_ppo = False
        return super().on_fit_start()

    def on_train_epoch_start(self) -> None:
        self.metric_tracker.reset()
        self.task_metrics.reset()

        if self.train_ppo:
            self.eval()
            self.env.reward_fn.fit_conformal_from_loader(batch_norm_fn=self.get_batch_norm)
            self.train()

        return super().on_train_epoch_start()

    def on_train_batch_start(self, batch: pygd.Batch, batch_idx: int):
        if self.train_ppo:
            self.eval()
            self.ppo.prepare_forward(env=self.env, n_steps=self.n_steps)

            self.train()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.train_ppo:
            self.ppo.forward_end()

    @property
    def automatic_optimization(self):
        return False

    # process inside the training loop
    def training_step(self, train_batch: pygd.Batch, batch_idx: int):
        tic = time.time()

        opt_clf, opt_policy = self.optimizers()

        if self.train_ppo:
            self.ppo.train()
            self.graph_clf.eval()
            tic_ppo = time.time()
            for k in range(self.ppo_steps):
                opt_policy.zero_grad()
                loss_dict = self.policy_loss_dict(shuffle=False)
                loss = loss_dict["ppo_loss"].mean()

                self.manual_backward(loss)
                opt_policy.step()
                for name, value in loss_dict.items():
                    self.metric_tracker.update(f"training/{name}", value.mean().item())

            self.metric_tracker.update("training/ppo_time", time.time() - tic_ppo)
        else:
            tic_clf = time.time()

            policy_kwargs = None
            if self.use_sparse_graph:
                policy_kwargs = {"sample": True, "num_samples": 1}

            self.ppo.eval()
            self.graph_clf.train()

            opt_clf.zero_grad()

            loss = self.graph_clf_loss(batch=train_batch, policy_kwargs=policy_kwargs)

            self.manual_backward(loss.mean())
            opt_clf.step()

            self.metric_tracker.update("training/graph_clf_loss", loss.item())

            self.metric_tracker.update("training/graph_clf_time", time.time() - tic_clf)

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

        if self.current_epoch > (self.trainer.max_epochs - 10):
            if not self.gnn_mode:
                self.train_ppo = True

        self.my_logger.track_value("training/train_ppo", int(self.train_ppo))
        self.my_logger.track_value("training/use_sparse_graph", int(self.use_sparse_graph))

        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.metric_tracker.reset(regex="valid")
        self.task_metrics.reset(regex="valid")
        return super().on_validation_epoch_start()

    def validation_step(self, batch: pygd.Batch, batch_idx: int):
        self.ppo.eval()
        self.graph_clf.eval()

        logits, target, stats = self.predict(batch, policy_kwargs=None)

        loss = self.loss_fn(logits, target)

        self.metric_tracker.update("valid/graph_clf_loss_full", loss.item())

        for name, value in stats.items():
            self.metric_tracker.update(f"valid/full__{name}", value)

        for key in self.task_metrics.get_keys(filter="valid.*full"):
            self.task_metrics.update(key, logits=logits, target=target)

        if self.use_sparse_graph:
            for num_samples in self.num_samples_list:
                logits, target, stats = self.predict(
                    batch, policy_kwargs={"sample": True, "num_samples": num_samples}
                )
                loss = self.loss_fn(logits, target)

                self.metric_tracker.update(f"valid/graph_clf_loss_{num_samples}", loss.item())

                for name, value in stats.items():
                    self.metric_tracker.update(f"valid/{name}__{num_samples}", value)

                for key in self.task_metrics.get_keys(filter=f"valid.*{num_samples}__"):
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

        if self.train_ppo:
            early_stopper_metric = metrics_dict[self.metric_ppo]
            if self.early_stopper_rl.should_stop(early_stopper_metric):
                self.early_stopper_clf.reset()
                self.early_stopper_rl.reset()

                self.train_ppo = False
                self.bilevel_iteration_count += 1

        else:
            early_stopper_metric = metrics_dict[self.metric_clf]
            cond_1 = (
                self.bilevel_iteration_count == 0
                and self.current_epoch > self.trainer.max_epochs * 0.85
            )
            cond_2 = self.early_stopper_clf.should_stop(early_stopper_metric)
            if cond_1 or cond_2:
                self.early_stopper_clf.reset()
                self.early_stopper_rl.reset()
                if not self.gnn_mode:
                    self.train_ppo = True
                    if not self.use_sparse_graph:
                        self.use_sparse_graph = True
                        self.metric_clf = self.metric_objective
                        self.early_stopper_clf.patience = self.early_stopper_rl.patience
                else:
                    self.trainer.should_stop = True

        if self.bilevel_iteration_count >= 4:
            self.trainer.should_stop = True

        return super().on_validation_epoch_end()

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
            "full": None,
            "1": {"sample": True, "num_samples": 1},
            # "10": {"sample": True, "num_samples": 10},
        }

        if plotter is not None and images_folder is not None:
            os.makedirs(images_folder, exist_ok=True)

            for policy_key in policy_dict.keys():
                if policy_key == "1":
                    output_metrics[f"{policy_key}__plot"] = []

        plot_count_max = 30

        plot_columns = ["data", "1__data_norm", "graph_id", "y", "1__y_pred"]
        plot_data = []

        self.to(device)

        for policy_key, policy_kwargs in policy_dict.items():
            task_metrics.reset()
            metric_tracker.reset()
            plot_count = 0
            for batch_id, batch in enumerate(loader):
                batch = batch.to(device)
                logits, target, stats, batch_norm = self.predict(
                    batch, policy_kwargs=policy_kwargs, return_batch_norm=True
                )

                target_pred = self.env.reward_fn.logits_to_hard_pred(logits)
                loss = self.loss_fn(logits, target)
                metric_tracker.update("loss", loss.item())

                for stat_key, stat_value in stats.items():
                    metric_tracker.update(stat_key, stat_value)
                for metric_name in task_metrics.get_keys():
                    task_metrics.update(metric_name, logits=logits, target=target)

                if policy_key == "1" and plotter is not None and images_folder is not None:
                    data_list = batch.to_data_list()
                    data_norm_list = batch_norm.to_data_list()
                    last_y = None
                    for data_id, (data_i, data_norm_i) in enumerate(zip(data_list, data_norm_list)):
                        y = int(data_i.y.item())
                        if last_y is None:
                            last_y = y
                        data_i = data_i.to("cpu")
                        action = data_norm_i.action.flatten().cpu().numpy()

                        node_color_attr_original = "node_color"
                        node_color_attr = "node_color"
                        edge_color_attr = None

                        if hasattr(data_i, "node_color"):
                            data_i.node_color += 2
                        else:
                            data_i.node_color = torch.zeros(data_i.num_nodes)

                        data_norm_i = data_i.clone()

                        graph_id = batch_id * loader.batch_size + data_id

                        if plot_count < plot_count_max and y != last_y:
                            last_y = y
                            if self.env.action_refers_to == cte.ActionTypes.NODE:
                                edge_attr = torch.arange(data_i.num_edges)
                                nodes_idx = np.where(action == 1)[0].tolist()
                                _, _, edge_attr_keep = PyGUtils.remove_nodes(
                                    num_nodes=data_i.num_nodes,
                                    edge_index=data_i.edge_index,
                                    nodes_idx=nodes_idx,
                                    edge_attr=edge_attr,
                                )

                                edge_color = torch.ones(data_i.num_edges) * 16
                                edge_color[edge_attr_keep] = 17
                                data_norm_i.node_color[action == 1] = 16
                                edge_color_attr = "edge_color"

                            elif self.env.action_refers_to == cte.ActionTypes.EDGE:
                                node_color_attr = None
                                edge_color = torch.ones(data_i.num_edges) * 16
                                edge_color[action == 1] = 17
                                edge_color_attr = "edge_color"

                            data_norm_i.edge_color = edge_color

                            file_path = os.path.join(
                                images_folder, f"{policy_key}__data_{batch_id}_{data_id}_{y}.png"
                            )
                            file_path_norm = os.path.join(
                                images_folder,
                                f"{policy_key}__data_norm_{batch_id}_{data_id}_{y}.png",
                            )
                            plotter.plot_graph(
                                graph=data_i,
                                show=False,
                                file_path=file_path,
                                node_color_attr=node_color_attr_original,
                            )
                            plotter.plot_graph(
                                graph=data_norm_i,
                                show=False,
                                file_path=file_path_norm,
                                node_color_attr=node_color_attr,
                                edge_color_attr=edge_color_attr,
                            )
                            plot_count += 1

                            y = target[data_id].item()
                            y_pred = target_pred[data_id].item()

                            plot_data.append([file_path, file_path_norm, graph_id, y, y_pred])

                            plotter.close_all()

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
        opt_ppo = {}

        # Set up Graph CLF
        params_clf = self.graph_clf.parameters()

        opt_clf["optimizer"] = TorchUtils.build_optimizer(
            name=self.optimizer_clf, optim_kwargs=self.optim_kwargs_clf, params=params_clf
        )

        if self.lr_scheduler_clf != cte.LRSchedulerType.NONE:
            opt_clf["lr_scheduler"] = TorchUtils.build_scheduler(
                name=self.lr_scheduler_clf,
                scheduler_kwargs=self.lr_scheduler_kwargs_clf,
                optimizer=opt_clf["optimizer"],
            )
            opt_clf["monitor"] = self.metric_objective

        # Set up Graph RL

        assert self.optim_kwargs_rl.ratio_critic >= 1.0
        assert self.optim_kwargs_rl.ratio_clf >= 1.0
        lr_base = self.optim_kwargs_rl.lr_clf / self.optim_kwargs_rl.ratio_clf
        lr_actor = lr_base
        lr_critic = lr_base / self.optim_kwargs_rl.ratio_critic
        assert self.optimizer_rl == cte.OptimizerType.ADAM
        params_policy = self.ppo.get_optimization_config(lr_actor=lr_actor, lr_critic=lr_critic)
        opt_ppo["optimizer"] = torch.optim.Adam(params_policy)
        if self.lr_scheduler_rl != cte.LRSchedulerType.NONE:
            opt_ppo["lr_scheduler"] = TorchUtils.build_scheduler(
                name=self.lr_scheduler_rl,
                scheduler_kwargs=self.lr_scheduler_kwargs_rl,
                optimizer=opt_ppo["optimizer"],
            )
            opt_ppo["monitor"] = self.metric_objective

        return (opt_clf, opt_ppo)

    def do_scheduler_step(self, sch: t_lr.LRScheduler, monitor: Optional[str], epoch_type: str):
        if epoch_type == "train":
            if isinstance(sch, list):
                for i, sch_i in enumerate(sch):
                    if i == 1 and not self.train_ppo:
                        continue
                    if not isinstance(sch_i, t_lr.ReduceLROnPlateau):
                        sch_i.step()
            elif sch is not None and not isinstance(sch, t_lr.ReduceLROnPlateau):
                sch.step()
        elif epoch_type == "val":
            if isinstance(sch, list):
                for i, sch_i in enumerate(sch):
                    if i == 1 and not self.train_ppo:
                        continue
                    if isinstance(sch_i, t_lr.ReduceLROnPlateau):
                        sch_i.step(monitor)
            elif sch is not None and isinstance(sch, t_lr.ReduceLROnPlateau):
                sch.step(monitor)
