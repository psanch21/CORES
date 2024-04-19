from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import torch_geometric.data as pygd
import torch_geometric.nn as geom_nn

from cores.core.values.constants import ActivationTypes, PoolingTypes
from cores.impl.activation.activation_torch import get_activation


class GraphPooling(nn.Module):
    """
    :param pool_type: (gym.Space)
    :param in_channels: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param out_channels: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        pool_type: List[PoolingTypes] | PoolingTypes = PoolingTypes.MEAN,
        in_channels: int = 256,
        activation: ActivationTypes = ActivationTypes.IDENTITY,
        out_channels: int = None,
        bn=False,
    ):
        super(GraphPooling, self).__init__()
        assert not bn

        logging.debug(f"pool_type: {type(pool_type)}")
        if not isinstance(pool_type, list):
            pool_type = [pool_type]

        pool_type = [PoolingTypes(pool_type_i) for pool_type_i in pool_type]

        self.pool_type = pool_type
        self.pool = None
        if PoolingTypes.GLOBAL_ATT in pool_type:  # Global Attention
            raise NotImplementedError
            # The output dim of gate_nn should be 1
            self.pool = geom_nn.aggr.AttentionalAggregation(
                gate_nn=nn.Linear(in_channels, 1), nn=None
            )
        # elif PoolingTypes.TOPK in pool_type:
        #     self.pool = geom_nn.TopKPooling(
        #         in_channels=in_channels,
        #         ratio=0.5,
        #         min_score=None,
        #         multiplier=1.0,
        #         nonlinearity=nn.Identity(),
        #     )

        if len(self.pool_type) > 1:
            assert in_channels % len(self.pool_type) == 0
            self.lin = nn.Sequential(
                get_activation(activation),
                nn.Linear(in_channels, in_channels // len(self.pool_type)),
            )
        else:
            self.lin = nn.Identity()

        if out_channels is not None:
            self.lin_out = nn.Sequential(
                get_activation(activation),
                nn.Linear(in_channels, out_channels),
            )
        else:
            self.lin_out = nn.Identity()

        self.use_bn = bn
        if bn:
            self.bn = nn.BatchNorm1d(num_features=in_channels)

    def forward(self, batch: pygd.Data) -> torch.FloatTensor:
        batch_ = batch.batch
        x = batch.x
        out = self.forward2(x=x, batch=batch_)
        return out

    def forward2(self, x: torch.FloatTensor, batch: pygd.Data) -> torch.FloatTensor:
        Z_global = []

        if PoolingTypes.MEAN in self.pool_type:
            Z_global_i = geom_nn.global_mean_pool(x=x, batch=batch)
            Z_global.append(Z_global_i)
        if PoolingTypes.MAX in self.pool_type:
            Z_global_i = geom_nn.global_max_pool(x=x, batch=batch)
            Z_global.append(Z_global_i)
        # if PoolingTypes.STD in self.pool_type:
        #     Z_global_i = global_std_pool(x=x, batch=batch)
        #     Z_global.append(Z_global_i)
        if PoolingTypes.ADD in self.pool_type:
            Z_global_i = geom_nn.global_add_pool(x=x, batch=batch)
            Z_global.append(Z_global_i)
        if self.pool is not None:
            if isinstance(self.pool, geom_nn.TopKPooling):
                outputs = self.pool(x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr)
                Z_global_i = outputs[0]
            else:
                Z_global_i = self.pool(x=x, batch=batch)

            Z_global.append(Z_global_i)

        assert len(Z_global) > 0

        if len(Z_global) > 1:
            Z_global_ = []
            for Z_i in Z_global:
                Z_global_.append(self.lin(Z_i))
            out = torch.cat(Z_global_, dim=1)
        else:
            z_pool = torch.cat(Z_global, dim=1)

            out = self.lin(z_pool)

        out = self.lin_out(out)

        if self.use_bn:
            out = self.bn(out)

        return out


def global_std_pool(x: torch.FloatTensor, batch: pygd.Data) -> torch.FloatTensor:
    unique_batches, counts = torch.unique(batch, return_counts=True)
    stds = []
    for batch_id in unique_batches:
        mask = batch == batch_id
        nodes_in_batch = mask.sum()
        if nodes_in_batch == 1:
            stds.append(torch.zeros_like(x[0]))
        else:
            stds.append(torch.std(x[mask], dim=0))
    return torch.stack(stds)
