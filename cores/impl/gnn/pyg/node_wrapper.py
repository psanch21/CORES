from __future__ import annotations

import torch.nn as nn
import torch_geometric.data as pygd


class NodeWrapper(nn.Module):
    def __init__(self, layer: nn.Module):
        super(NodeWrapper, self).__init__()

        self.layer = layer

    def forward(self, batch: pygd.Data, **kwargs):
        batch.x = self.layer(batch.x, *kwargs)
        return batch
