from __future__ import annotations

import argparse
import json
import logging
import os

import omegaconf
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import cores.core.values.constants as cte
import cores.impl.dataset_preparators as gdpp
import cores.impl.gnn.pyg as ggnn
import cores.impl.logger as mlul
import cores.impl.metrics as gmet
import cores.impl.rl.graph.envs as genv
import cores.impl.rl.graph.policy as gpol
import cores.impl.rl.graph.rewards as grr
from cores.impl.rl.graph.ppo import GraphPPO


class Provider:
    @staticmethod
    def dataset_preparator(name: cte.Datasets, **kwargs):
        if name == cte.Datasets.BA2MOTIFS:
            dataset_preparator = gdpp.BA2MotifPreparator(**kwargs)
        elif name == cte.Datasets.BZR:
            dataset_preparator = gdpp.BZRPreparator(**kwargs)
        elif name == cte.Datasets.COX2:
            dataset_preparator = gdpp.COX2Preparator(**kwargs)
        elif name == cte.Datasets.DD:
            dataset_preparator = gdpp.DDPreparator(**kwargs)
        elif name == cte.Datasets.ENZYMES:
            dataset_preparator = gdpp.ENZYMESPreparator(**kwargs)
        elif name == cte.Datasets.MUTAG:
            dataset_preparator = gdpp.MUTAGPreparator(**kwargs)
        elif name == cte.Datasets.NCI1:
            dataset_preparator = gdpp.NCI1Preparator(**kwargs)
        elif name == cte.Datasets.NCI109:
            dataset_preparator = gdpp.NCI109Preparator(**kwargs)
        elif name == cte.Datasets.PROTEINS:
            dataset_preparator = gdpp.ProteinsPreparator(**kwargs)
        elif name == cte.Datasets.PTC_FM:
            dataset_preparator = gdpp.PTCFMPreparator(**kwargs)
        else:
            raise ValueError(f"Invalid dataset preparator {name}")

        return dataset_preparator

    @staticmethod
    def graph_env(name: cte.GraphEnvTypes, **kwargs) -> genv.GraphEnvBase:
        if name == cte.GraphEnvTypes.SINGLE:
            env_graph = genv.GraphEnvOne(**kwargs)
        elif name == cte.GraphEnvTypes.MULTI:
            env_graph = genv.GraphEnv(**kwargs)
        else:
            raise ValueError(f"Invalid graph enviroment {name}")

        return env_graph

    @staticmethod
    def init_fn(name: cte.InitFnTypes, **kwargs) -> nn.Module:
        if name == cte.InitFnTypes.XAVIER:

            def init_fn(m):
                """Performs weight initialization."""
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1.0)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data = nn.init.xavier_uniform_(
                        m.weight.data, gain=nn.init.calculate_gain(**kwargs)
                    )
                    if m.bias is not None:
                        m.bias.data.zero_()

        elif name == cte.InitFnTypes.NORMAL:

            def init_fn(module):
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, **kwargs)
                    if isinstance(module.bias, torch.Tensor):
                        nn.init.zeros_(module.bias)

        else:
            raise ValueError(f"Invalid init function {name}")

        return init_fn

    @staticmethod
    def gnn(name: cte.GNNLayers, **kwargs) -> ggnn.BaseGNN:
        if name == cte.GNNLayers.GCN:
            gnn = ggnn.GCN(**kwargs)
        elif name == cte.GNNLayers.GAT:
            gnn = ggnn.GAT(**kwargs)
        elif name == cte.GNNLayers.GIN:
            gnn = ggnn.GIN(**kwargs)
        else:
            raise ValueError(f"Invalid GNN {name}")

        return gnn

    @staticmethod
    def loss_fn(name: cte.LossType, **kwargs) -> nn.Module:
        if name == cte.LossType.CROSS_ENTROPY:
            loss_fn = nn.CrossEntropyLoss(**kwargs)
        # elif name == cte.LossType.BCE:
        #     loss_fn = nn.BCELoss(**kwargs)
        elif name == cte.LossType.BCE_LOGITS:
            loss_fn = nn.BCEWithLogitsLoss(**kwargs)
        else:
            raise ValueError(f"Invalid loss function {name}")

        return loss_fn

    @staticmethod
    def logger(name: cte.LoggerType, **kwargs) -> cte.LoggerType:
        if name == cte.LoggerType.DUMMY:
            logger = mlul.DummyLogger()
        elif name == cte.LoggerType.FILE_SYSTEM:
            logger = mlul.FSLogger(**kwargs)
        elif name == cte.LoggerType.PRINT:
            logger = mlul.PrintLogger(**kwargs)
        elif name == cte.LoggerType.WANDB:
            logger = mlul.WandBLogger(**kwargs)
        else:
            raise ValueError(f"Invalid logger {name}")

        return logger

    @staticmethod
    def metrics(name: cte.TaskTypes, **kwargs) -> cte.LoggerType:
        if name == cte.TaskTypes.CLF_BINARY:
            metrics = gmet.BinaryCLFMetricsTorch(**kwargs)
        elif name == cte.TaskTypes.CLF_MULTICLASS:
            metrics = gmet.MultiCLFMetricsTorch(**kwargs)
        else:
            raise ValueError(f"Invalid metric {name}")

        return metrics

    @staticmethod
    def parser(description: str, config_file_default: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "--config_file",
            type=str,
            default=config_file_default,
            help="Path to the configuration file",
        )

        parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)

        return parser

    @staticmethod
    def cfg(
        config_file_default="config/cores/train.yaml",
        config_file_dict=None,
        description="Default description",
    ):
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "--config_file",
            type=str,
            default=config_file_default,
            help="Path to the configuration file",
        )

        if config_file_dict is not None:
            for key, value in config_file_dict.items():
                parser.add_argument(
                    f"--{key}",
                    type=str,
                    default=value,
                    help=f"Path to the configuration file for {key}",
                )
        parser.add_argument(
            "--config_extra",
            type=str,
            default=None,
            help="Path to the extra json config file",
        )

        parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)

        args = parser.parse_args()

        cfg = omegaconf.OmegaConf.load(args.config_file)

        if config_file_dict is not None:
            for key in config_file_dict:
                config_file_i = getattr(args, key)
                if config_file_i is not None:
                    cfg_i = omegaconf.OmegaConf.load(config_file_i)
                    cfg = OmegaConf.unsafe_merge(cfg, cfg_i)

        if args.config_extra is not None:
            config_exta_files = args.config_extra.split("+")
            for config_extra in config_exta_files:
                # Check extension of the file
                extension = os.path.splitext(config_extra)[1]
                opts_list = []

                if extension == ".json":
                    logging.info(f"Loading extra config file: {config_extra}...")
                    config = json.load(open(config_extra))

                    for key, key_dict in config.items():
                        if key == "_wandb":
                            continue
                        value = key_dict["value"]
                        opts_list.append(f"{key}={value}")

                elif extension == ".yaml":
                    logging.info(f"Loading extra config file: {config_extra}...")
                    config = OmegaConf.load(config_extra)

                    for key, value in config.items():
                        opts_list.append(f"{key}={value}")

                cfg.merge_with_dotlist(opts_list)

        if args.opts:
            cfg.merge_with_dotlist(args.opts)

        return cfg

    @staticmethod
    def policy(name: cte.PolicyTypes, **kwargs) -> nn.Module:
        if name == cte.PolicyTypes.GRAPH_ACTOR_CRITIC:
            policy = gpol.GraphActorCritic(**kwargs)
        else:
            raise ValueError(f"Invalid policy {name}")

        return policy

    @staticmethod
    def ppo_graph(**kwargs) -> GraphPPO:
        ppo = GraphPPO(**kwargs)

        return ppo

    @staticmethod
    def reward_fn(name: cte.RewardTypes, **kwargs) -> nn.Module:
        if name == cte.RewardTypes.CORES_CONFORMAL:
            reward = grr.RewardConformal(**kwargs)
        else:
            raise ValueError(f"Invalid reward function {name}")

        return reward
