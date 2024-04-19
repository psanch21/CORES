from __future__ import annotations

import random

import torch_geometric.nn as pygnn

from cores.impl.gnn.pyg.gnn_base import BaseGNN


class MyGCNConv(pygnn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(MyGCNConv, self).__init__(*args, **kwargs)

    def forward(self, batch):
        x = super(MyGCNConv, self).forward(x=batch.x, edge_index=batch.edge_index, edge_weight=None)

        batch.x = x
        return batch


class GCN(BaseGNN):
    @staticmethod
    def random_kwargs(seed):
        kwargs = BaseGNN.random_kwargs(seed)

        kwargs["improved"] = random.choice([True, False])

        return kwargs

    def __init__(self, *args, improved: bool = False, **kwargs):
        self.improved = improved

        super(GCN, self).__init__(*args, **kwargs)

    def _gnn_layer(self, input_dim: int, output_dim: int) -> MyGCNConv:
        return MyGCNConv(
            in_channels=input_dim,
            out_channels=output_dim,
            improved=self.improved,
            cached=False,
            add_self_loops=True,
            normalize=True,
            bias=True,
        )
