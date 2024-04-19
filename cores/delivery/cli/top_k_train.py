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
from cores.impl.top_k import TopKLightning
from cores.provider import Provider
from cores.utils.plotter import MatplotlibPlotter

# Argsparse


def train_topk(cfg):
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

    init_fn = Provider.init_fn(cte.InitFnTypes(cfg.init_fn), **cfg.init_fn_kwargs)

    metrics_kwargs = OmegaConf.to_object(cfg.metrics_kwargs)
    metrics_kwargs["classes_num"] = preparator.classes_num()

    metrics = Provider.metrics(cte.TaskTypes(cfg.metrics), **metrics_kwargs)

    top_k_kwargs = {}

    if cfg.model == "top_k":
        if cfg.model_kwargs.use_gnn:
            logging.info(f"Loading GNN: {cfg.model_kwargs.use_gnn}...")
            gnn_kwargs = OmegaConf.to_object(cfg.graph_clf_kwargs)

            gnn_kwargs["input_dim"] = preparator.features_dim()
            gnn_kwargs["output_dim"] = preparator.features_dim()
            gnn_kwargs["hidden_dim"] = preparator.features_dim()
            gnn_kwargs["layers_pre_num"] = 0
            gnn_kwargs["layers_gnn_num"] = 1
            gnn_kwargs["layers_post_num"] = 0
            gnn_kwargs["pooling"] = None

            gnn = Provider.gnn(cte.GNNLayers(cfg.graph_clf), **gnn_kwargs)
        else:
            gnn = None

        top_k_kwargs["gnn"] = gnn
        top_k_kwargs["ratio"] = cfg.model_kwargs.ratio
        top_k_kwargs["min_score"] = cfg.model_kwargs.min_score
        top_k_kwargs["multiplier"] = cfg.model_kwargs.multiplier

    elif cfg.model == "g_pool":
        top_k_kwargs["k"] = cfg.model_kwargs.k
        top_k_kwargs["p"] = cfg.model_kwargs.p
    model_lit = TopKLightning(
        graph_clf=graph_clf,
        loss_fn=loss_fn,
        optimizer=cte.OptimizerType(cfg.optimizer),
        optimizer_kwargs=cfg.optimizer_kwargs,
        lr_scheduler=cte.LRSchedulerType(cfg.lr_scheduler),
        lr_scheduler_kwargs=cfg.lr_scheduler_kwargs,
        early_stopping_kwargs=cfg.early_stopping_kwargs,
        metric_objective=cfg.model_kwargs.metric_objective.name,
        top_k_mode=cfg.model,
        top_k_kwargs=top_k_kwargs,
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

        # Save checkpoint
        task_metrics = Provider.metrics(cte.TaskTypes(cfg.metrics), **metrics_kwargs)

        # ckpt_dict = {"best": ckpt_best, "end": ckpt_end}
        ckpt_dict = {"end": ckpt_end}
        # ckpt_dict = {"best": ckpt_best}

        plotter = MatplotlibPlotter()

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

    config_file_dict["config_file_data"] = os.path.join("config", "dataset", "mutag.yaml")
    config_file_dict["config_file_machine"] = os.path.join("config", "machine", "laptop.yaml")
    config_file_dict["config_file_logger"] = os.path.join("config", "logger", "file_system.yaml")

    config_file_dict["config_file_top_k"] = os.path.join("config", "top_k", "top_k_soft.yaml")

    cfg = Provider.cfg(
        description="Train TopK model.",
        config_file_default="config/top_k/train.yaml",
        config_file_dict=config_file_dict,
    )

    train_topk(cfg)
