from __future__ import annotations

import random
from typing import Optional

import torch.nn as nn
import torch_geometric.nn as pygnn

from cores.impl.activation.activation_torch import get_activation
from cores.impl.gnn.pyg.gnn_base import BaseGNN


class MyGINEConv(pygnn.GINEConv):
    def __init__(self, *args, **kwargs):
        super(MyGINEConv, self).__init__(*args, **kwargs)

    def forward(self, batch):
        x = super(MyGINEConv, self).forward(
            x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
        )

        batch.x = x
        return batch


class GINE(BaseGNN):
    @staticmethod
    def random_kwargs(seed):
        kwargs = BaseGNN.random_kwargs(seed)

        kwargs["eps"] = random.choice([0.0, 0.1, 0.2, 0.3])
        kwargs["train_eps"] = random.choice([True, False])

        return kwargs

    def __init__(
        self,
        *args,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        self.eps = eps
        self.train_eps = train_eps
        self.edge_dim = edge_dim
        super(GINE, self).__init__(*args, **kwargs)

    def _gnn_layer(self, input_dim: int, output_dim: int) -> MyGINEConv:
        layers = [nn.Linear(input_dim, output_dim)]
        if self.has_bn:
            layers.append(nn.BatchNorm1d(output_dim))
        layers.append(get_activation(self.activation))
        layers.append(nn.Linear(output_dim, output_dim))

        net = nn.Sequential(*layers)
        return MyGINEConv(
            nn=net, eps=self.eps, train_eps=self.train_eps, edge_dim=self.edge_dim, aggr="add"
        )
