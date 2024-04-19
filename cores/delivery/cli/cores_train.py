import logging
import os
import sys

import lightning as L
import lightning.pytorch.callbacks as lpcall
import numpy as np
import torch
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import cores.core.values.constants as cte
from cores.impl.cores import CORESLightning
from cores.provider import Provider
from cores.utils.plotter import MatplotlibPlotter

# Argsparse


def train_cores(cfg):
    # Set seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    logging.info(f"Loader dataset preparator: {cfg.data}...")
    preparator = Provider.dataset_preparator(cfg.data, **cfg.data_kwargs)

    preparator.prepare()

    cfg.graph_clf_kwargs.input_dim = preparator.features_dim()
    cfg.graph_clf_kwargs.output_dim = preparator.target_dim()

    logger_kwargs = OmegaConf.to_object(cfg.logger_kwargs)

    assert "config" not in logger_kwargs

    config = OmegaConf.to_object(cfg)

    # Remove logger and logger_kwargs
    config.pop("logger")
    config.pop("logger_kwargs")

    logger_kwargs["config"] = config

    logger = Provider.logger(cte.LoggerType(cfg.logger), **logger_kwargs)

    loaders = preparator.get_dataloaders(**cfg.dataloader_kwargs)

    logging.info(f"Loading Graph CLF: {cfg.graph_clf}...")

    gnn_kwargs = OmegaConf.to_object(cfg.graph_clf_kwargs)

    graph_clf = Provider.gnn(cte.GNNLayers(cfg.graph_clf), **gnn_kwargs)

    logging.info(f"Loading loss: {cfg.loss}...")

    loss_fn = Provider.loss_fn(cte.LossType(cfg.loss), **cfg.loss_kwargs)

    logging.info(f"Loading Reward: {cfg.reward}...")

    reward_kwargs = OmegaConf.to_object(cfg.reward_kwargs)

    reward_kwargs["graph_clf"] = graph_clf
    reward_kwargs["classes_num"] = preparator.classes_num()
    reward_kwargs["loss_fn"] = loss_fn
    reward_kwargs["action_refers_to"] = cte.ActionTypes(cfg.cores.action_refers_to)

    reward_type = cte.RewardTypes(cfg.reward)

    if reward_type == cte.RewardTypes.CORES_CONFORMAL:
        reward_kwargs["loader_calib"] = preparator.get_dataloader("train", **cfg.dataloader_kwargs)

    reward_fn = Provider.reward_fn(cte.RewardTypes(cfg.reward), **reward_kwargs)

    # if reward_type == cte.RewardTypes.CORES_CONFORMAL:
    #     reward_fn.fit_conformal_from_loader()

    logging.info(f"Loading GraphEnv: {cfg.graph_env}...")

    graph_env_kwargs = OmegaConf.to_object(cfg.graph_env_kwargs)
    graph_env_kwargs["loader"] = preparator.get_dataloader("valid", batch_size=1, shuffle=False)
    graph_env_kwargs["graph_clf"] = graph_clf
    graph_env_kwargs["reward_fn"] = reward_fn
    graph_env_kwargs["action_refers_to"] = cte.ActionTypes(cfg.cores.action_refers_to)
    env = Provider.graph_env(cte.GraphEnvTypes(cfg.graph_env), **graph_env_kwargs)

    policy_gnn_kwargs = OmegaConf.to_object(cfg.graph_clf_kwargs)
    policy_gnn_kwargs["pooling"] = None
    policy_gnn_kwargs["output_dim"] = cfg.graph_clf_kwargs.hidden_dim

    policy_gnn = Provider.gnn(cte.GNNLayers(cfg.graph_clf), **policy_gnn_kwargs)

    policy_kwargs = OmegaConf.to_object(cfg.policy_kwargs)
    policy_kwargs["gnn"] = policy_gnn
    policy_kwargs["action_refers_to"] = cte.ActionTypes(cfg.cores.action_refers_to)
    policy_kwargs["action_distr"] = env.get_action_distr_name()
    policy = Provider.policy(cte.PolicyTypes(cfg.policy), **policy_kwargs)

    ppo_kwargs = OmegaConf.to_object(cfg.ppo_kwargs)

    ppo_kwargs["policy"] = policy

    ppo_graph = Provider.ppo_graph(**ppo_kwargs)

    init_fn = Provider.init_fn(cte.InitFnTypes(cfg.init_fn), **cfg.init_fn_kwargs)

    metrics_kwargs = OmegaConf.to_object(cfg.metrics_kwargs)
    metrics_kwargs["classes_num"] = preparator.classes_num()

    metrics = Provider.metrics(cte.TaskTypes(cfg.metrics), **metrics_kwargs)

    model_lit = CORESLightning(
        env=env,
        graph_clf=graph_clf,
        ppo=ppo_graph,
        loss_fn=loss_fn,
        optimizer_clf=cte.OptimizerType(cfg.optimizer_clf),
        optimizer_kwargs_clf=cfg.optimizer_kwargs_clf,
        lr_scheduler_clf=cte.LRSchedulerType(cfg.lr_scheduler_clf),
        lr_scheduler_kwargs_clf=cfg.lr_scheduler_kwargs_clf,
        optimizer_rl=cte.OptimizerType(cfg.optimizer_rl),
        optimizer_kwargs_rl=cfg.optimizer_kwargs_rl,
        lr_scheduler_rl=cte.LRSchedulerType(cfg.lr_scheduler_rl),
        lr_scheduler_kwargs_rl=cfg.lr_scheduler_kwargs_rl,
        env_steps=cfg.cores.env_steps,
        ppo_steps=cfg.cores.ppo_steps,
        early_stopping_kwargs=cfg.early_stopping_kwargs,
        metric_objective=cfg.cores.metric_objective,
        gnn_mode=cfg.cores.gnn_mode,
        init_fn=init_fn,
    )

    model_lit.set_task_metrics(metrics)
    model_lit.set_my_logger(logger)

    callbacks = []

    if cfg.checkpoint == "enabled":
        checkpoint = lpcall.ModelCheckpoint(
            dirpath=logger.folder(),
            **cfg.checkpoint_kwargs,
        )
        callbacks.append(checkpoint)
    trainer = L.Trainer(callbacks=callbacks, **cfg.trainer)

    trainer.fit(
        model=model_lit, train_dataloaders=loaders["train"], val_dataloaders=loaders["valid"]
    )

    if cfg.checkpoint == "enabled":
        ckpt_best = checkpoint.best_model_path
        ckpt_folder = os.path.dirname(ckpt_best)
        ckpt_end = os.path.join(ckpt_folder, "end.ckpt")
        trainer.save_checkpoint(ckpt_end)

        # ckpt_dict = {"best": ckpt_best, "end": ckpt_end}
        ckpt_dict = {"end": ckpt_end}
        # ckpt_dict = {"best": ckpt_best}
    else:
        ckpt_dict = {"end": None}

    task_metrics = Provider.metrics(cte.TaskTypes(cfg.metrics), **metrics_kwargs)

    if cfg.plot:
        plotter = MatplotlibPlotter()
    else:
        plotter = None

    for ckpt_id, ckpt_i in ckpt_dict.items():
        print(f"\nLoading model: {ckpt_i}")

        metrics_testing = {}

        for split_name, loader in loaders.items():
            key_outer = f"testing_{split_name}_{ckpt_id}"

            images_folder = os.path.join(logger.folder(), f"images", f"{split_name}_{ckpt_id}")

            if not os.path.exists(images_folder):
                os.makedirs(images_folder)
            output_metrics, plot_dict = model_lit.evaluate(
                loader=loader,
                ckpt_file=ckpt_i,
                task_metrics=task_metrics,
                plotter=plotter,
                images_folder=images_folder,
                device=cfg.device,
            )

            for key, value in output_metrics.items():
                metrics_testing[f"{key_outer}/{key}"] = value

            logger.track_table(
                key=f"{key_outer}/data",
                columns=plot_dict["columns"],
                data=plot_dict["data"],
            )

        logger.track_data(metrics_testing)

    logger.finish()
    print(f"Folder: {logger.folder()}")


if __name__ == "__main__":
    config_file_dict = {}
    config_file_dict["config_file_reward"] = "config/cores/reward_conformal.yaml"
    os.path.join("config", "cores", "standard.yaml")
    config_file_dict["config_file_data"] = os.path.join("config", "dataset", "mutag.yaml")
    config_file_dict["config_file_machine"] = os.path.join("config", "machine", "laptop.yaml")
    config_file_dict["config_file_logger"] = os.path.join("config", "logger", "wandb.yaml")

    cfg = Provider.cfg(
        description="Train CORES model.",
        config_file_default="config/cores/train.yaml",
        config_file_dict=config_file_dict,
    )

    train_cores(cfg)
