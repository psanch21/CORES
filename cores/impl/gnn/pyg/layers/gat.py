from __future__ import annotations

import random
from typing import Optional

import torch_geometric.data as pygd
import torch_geometric.nn as pygnn

from cores.impl.gnn.pyg.gnn_base import BaseGNN


class MyGATConv(pygnn.GATConv):
    def __init__(self, *args, **kwargs):
        super(MyGATConv, self).__init__(*args, **kwargs)

    def forward(self, batch: pygd.Data):
        x = super(MyGATConv, self).forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            return_attention_weights=None,
        )

        batch.x = x
        return batch


class GAT(BaseGNN):
    @staticmethod
    def random_kwargs(seed):
        kwargs = BaseGNN.random_kwargs(seed)

        kwargs["heads"] = random.choice([1, 2, 4])

        return kwargs

    def __init__(self, *args, heads: int = 1, edge_dim: Optional[int] = None, **kwargs):
        self.heads = heads
        self.edge_dim = edge_dim
        super(GAT, self).__init__(*args, **kwargs)

    def _gnn_layer(self, input_dim: int, output_dim: int) -> MyGATConv:
        assert output_dim % self.heads == 0
        out_channels = output_dim // self.heads
        return MyGATConv(
            in_channels=input_dim,
            out_channels=out_channels,
            heads=self.heads,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
            edge_dim=self.edge_dim,
            fill_value="mean",
            bias=True,
        )
