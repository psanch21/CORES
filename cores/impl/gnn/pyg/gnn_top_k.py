from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cores.impl.gnn.pyg.gnn_base import BaseGNN
from cores.impl.gnn.pyg.layers.my_topk_pool import MyTopKPooling
from cores.utils import PyGUtils


class TopKGNN(nn.Module):
    @staticmethod
    def random_kwargs(seed):
        np.random.seed(seed)
        random.seed(seed)
        kwargs = {}

        kwargs["ratio"] = random.choice([0.2, 0.5, 1.0])
        kwargs["min_score"] = random.choice([None, -1.0, -0.5, 0.0, 0.3, 1.0])
        kwargs["multiplier"] = random.choice([1.0])
        return kwargs

    def __init__(
        self,
        in_channels: int,
        gnn: Optional[BaseGNN] = None,
        ratio: float = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        super(TopKGNN, self).__init__()

        self.gnn = gnn

        self.ratio = ratio
        if isinstance(min_score, float) and min_score == 0.0:
            min_score = None
        self.min_score = min_score
        self.multiplier = multiplier

        self.pool = MyTopKPooling(
            in_channels=in_channels,
            ratio=self.ratio,
            min_score=self.min_score,
            multiplier=self.multiplier,
            nonlinearity=torch.tanh,
        )

    def forward(self, batch, inplace=False, **kwargs):
        if self.gnn is None:
            logits = None
        else:
            logits = self.gnn(batch.clone())

        if inplace:
            batch_ = batch
        else:
            batch_ = batch.clone()

        perm, score = self.pool(
            x=batch_.x,
            edge_index=batch_.edge_index,
            batch=batch_.batch,
            edge_attr=batch_.edge_attr,
            attn=logits,
        )

        batch_ = PyGUtils.remove_nodes_from_batch_2(
            batch=batch,
            nodes_idx=perm.tolist(),
            mode="keep",
            relabel_nodes=True,
        )

        batch_.x *= score.view(-1, 1) * self.pool.multiplier

        return batch_
