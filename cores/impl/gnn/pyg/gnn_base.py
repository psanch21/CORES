from __future__ import annotations
import random
from abc import abstractmethod
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as pygd

from cores.core.values.constants import ActivationTypes, PoolingTypes, StageTypes
from cores.impl.activation.activation_torch import get_activation
from cores.impl.gnn.pyg.node_wrapper import NodeWrapper
from cores.impl.gnn.pyg.pooling import GraphPooling
from cores.impl.mlp.mlp_torch import MLP


class BaseGNN(nn.Module):
    @staticmethod
    def random_kwargs(seed):
        np.random.seed(seed)
        random.seed(seed)
        kwargs = {}
        kwargs["hidden_dim"] = random.choice([8, 16, 32, 64, 128])
        kwargs["output_dim"] = random.choice([8, 16, 32, 64, 128])
        kwargs["activation"] = random.choice(list(ActivationTypes))
        kwargs["dropout"] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        kwargs["has_bn"] = random.choice([True, False])
        kwargs["stage_type"] = random.choice(list(StageTypes))

        kwargs["layers_pre_num"] = np.random.randint(0, 4)
        kwargs["layers_gnn_num"] = np.random.randint(0, 4)
        kwargs["layers_post_num"] = np.random.randint(1, 3)

        # Remove PoolingTypes.GLOBAL_ATT from pooling_list
        pooling_list = [None, *list(PoolingTypes)]
        pooling_list.remove(PoolingTypes.GLOBAL_ATT)

        kwargs["pooling"] = random.choice(pooling_list)

        return kwargs

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: ActivationTypes,
        dropout: float,
        has_bn: bool,
        stage_type: StageTypes,
        layers_pre_num: int,
        layers_gnn_num: int,
        layers_post_num: int,
        pooling: Optional[PoolingTypes | List[PoolingTypes]] = None,
        device: str = "cpu",
    ):
        super(BaseGNN, self).__init__()

        if isinstance(activation, str):
            activation = ActivationTypes(activation)

        self.activation = activation
        self.device = device
        self.has_pre = layers_pre_num > 0
        self.has_gnn = layers_gnn_num > 0
        self.has_post = layers_post_num > 0

        self.layers_pre_num = layers_pre_num
        self.layers_gnn_num = layers_gnn_num
        self.layers_post_num = layers_post_num

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.has_bn = has_bn
        self.stage_type = stage_type
        self.lin_skipsum = None

        if self.has_pre:
            pre_input_dim = input_dim
            pre_output_dim = hidden_dim
        else:
            pre_input_dim = input_dim
            pre_output_dim = input_dim

        self.pre_nn = MLP(
            pre_input_dim,
            hidden_dim=hidden_dim,
            output_dim=pre_output_dim,
            layers_num=layers_pre_num,
            activation=activation,
            has_bn=has_bn,
            use_act_out=self.has_gnn or self.has_post,
            dropout=dropout,
            device=self.device,
        )

        if layers_post_num > 0:
            gnn_output_dim = hidden_dim
        else:
            gnn_output_dim = output_dim

        self.gnn = self._build_gnn(
            input_dim=self.pre_nn.output_dim,
            hidden_dim=hidden_dim,
            output_dim=gnn_output_dim,
            activation=activation,
            dropout=dropout,
            layers_num=layers_gnn_num,
            act_last=layers_post_num > 0,
        )

        if layers_gnn_num > 0:
            input_dim_post = hidden_dim
        else:
            input_dim_post = self.pre_nn.output_dim

        output_dim_post = hidden_dim if pooling is not None else output_dim

        self.post_nn = MLP(
            input_dim=input_dim_post,
            hidden_dim=hidden_dim,
            output_dim=output_dim_post,
            layers_num=layers_post_num,
            activation=activation,
            has_bn=has_bn,
            dropout=dropout,
            drop_last=True,
            device=self.device,
        )

        if pooling is not None:
            self.pooling = GraphPooling(
                pool_type=pooling,
                in_channels=output_dim_post,
                activation=activation,
                out_channels=output_dim,
                bn=False,
            )
        else:
            self.pooling = pooling

    @abstractmethod
    def forward(self, batch: pygd.Batch, **kwargs) -> torch.FloatTensor:

        batch = self.forward_pre(batch)

        batch = self.forward_gnn(batch, **kwargs)
        batch = self.forward_post(batch)

        if self.pooling is None:
            logits = batch.x
        else:
            logits = self.pooling(batch)
        return logits

    def forward_pre(self, batch: pygd.Batch) -> pygd.Data:

        x = self.pre_nn(batch.x)

        batch.x = x

        return batch

    @abstractmethod
    def _gnn_layer(self, input_dim: int, output_dim: int):
        pass

    def _build_gnn(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: ActivationTypes,
        dropout: float,
        layers_num: int,
        act_last: bool,
    ) -> nn.ModuleList:
        assert layers_num >= 0
        act_fn = NodeWrapper(get_activation(activation))

        layers = []

        if self.stage_type == StageTypes.SKIPSUM:
            linears = []
        for n in range(layers_num):
            input_dim_i = None
            output_dim_i = None
            act_i = None

            if n == 0:
                if layers_num == 1 and not act_last:
                    input_dim_i = input_dim
                    output_dim_i = output_dim
                    act_i = NodeWrapper(nn.Identity())
                else:
                    input_dim_i = input_dim
                    output_dim_i = hidden_dim
                    act_i = act_fn
            elif n == (layers_num - 1) and not act_last:
                input_dim_i = hidden_dim
                output_dim_i = hidden_dim
                act_i = NodeWrapper(nn.Identity())
            else:
                input_dim_i = hidden_dim
                output_dim_i = hidden_dim
                act_i = act_fn

            if self.stage_type == StageTypes.SKIPSUM:
                tmp = [nn.Linear(input_dim_i, output_dim)]
                if dropout > 0.0:
                    tmp.append(nn.Dropout(dropout))
                linears.append(nn.Sequential(*tmp))

            gnn_layer = self._gnn_layer(input_dim=input_dim_i, output_dim=output_dim_i)
            layers_i = [gnn_layer]
            if self.has_bn:
                layers_i.append(NodeWrapper(nn.BatchNorm1d(output_dim_i)))

            layers_i.append(act_i)
            if dropout > 0.0:
                layers_i.append(NodeWrapper(nn.Dropout(dropout)))

            layers.append(nn.Sequential(*layers_i))

        if layers_num == 0:
            layers = [nn.Identity()]

        if self.stage_type == StageTypes.SKIPSUM:
            self.lin_skipsum = nn.ModuleList(linears)

        return nn.ModuleList(layers)

    def forward_gnn(self, batch: pygd.Batch, **kwargs):
        out = 0.0
        if self.layers_gnn_num == 0:
            return batch
        for i, l in enumerate(self.gnn):
            if self.stage_type == StageTypes.SKIPSUM:
                out += self.lin_skipsum[i](batch.x)

            batch = l(batch, **kwargs)

        if self.stage_type == StageTypes.SKIPSUM:
            out += batch.x
            batch.x = out
        return batch

    def forward_post(self, batch: pygd.Batch) -> pygd.Batch:
        x = self.post_nn(batch.x)
        batch.x = x
        return batch
