from __future__ import annotations

import random

import torch.nn as nn
import torch_geometric.nn as pygnn

from cores.impl.activation.activation_torch import get_activation
from cores.impl.gnn.pyg.gnn_base import BaseGNN


class MyGINConv(pygnn.GINConv):
    def __init__(self, *args, **kwargs):
        super(MyGINConv, self).__init__(*args, **kwargs)

    def forward(self, batch):
        x = super(MyGINConv, self).forward(x=batch.x, edge_index=batch.edge_index)

        batch.x = x
        return batch


class GIN(BaseGNN):
    @staticmethod
    def random_kwargs(seed):
        kwargs = BaseGNN.random_kwargs(seed)

        kwargs["eps"] = random.choice([0.0, 0.1, 0.2, 0.3])
        kwargs["train_eps"] = random.choice([True, False])

        return kwargs

    def __init__(self, *args, eps: float = 0.0, train_eps: bool = False, **kwargs):
        self.eps = eps
        self.train_eps = train_eps
        super(GIN, self).__init__(*args, **kwargs)

    @staticmethod
    def kwargs(cfg, preparator):
        my_dict = {}
        my_dict["eps"] = cfg.layer.eps
        my_dict["train_eps"] = cfg.layer.train_eps
        my_dict.update(BaseGNN.kwargs(cfg, preparator))

        return my_dict

    def _gnn_layer(self, input_dim: int, output_dim: int) -> MyGINConv:
        layers = [nn.Linear(input_dim, output_dim)]
        if self.has_bn:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.append(get_activation(self.activation))
        layers.append(nn.Linear(output_dim, output_dim))

        net = nn.Sequential(*layers)
        return MyGINConv(nn=net, eps=self.eps, train_eps=self.train_eps, aggr="add")
