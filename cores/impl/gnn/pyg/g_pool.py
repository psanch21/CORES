from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch_geometric.data.batch as pygb


def top_k_graph(scores: torch.Tensor, g: torch.Tensor, h: torch.Tensor, k: int):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    return un_g, new_h, idx


class GPool(nn.Module):
    @staticmethod
    def random_kwargs(seed):
        random.seed(seed)
        kwargs = {}

        kwargs["k"] = random.choice([0.1, 0.3, 0.5, 0.7])
        kwargs["p"] = random.choice([0.0, 0.1, 0.2])
        return kwargs

    def __init__(self, in_channels: int, k: float, p: float, device: str = "cpu"):
        super(GPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_channels, 1)
        self.device = device
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

        self.sigmoid = self.sigmoid.to(self.device)
        self.proj = self.proj.to(self.device)
        self.drop = self.drop.to(self.device)

    def forward(self, batch: pygb.Batch):
        batch_ = batch.clone()
        data_list_out = []
        for graph in batch_.to_data_list():
            h = graph.x
            g = torch.zeros((h.shape[0], h.shape[0]), device=self.device)
            g[graph.edge_index[0], graph.edge_index[1]] = 1
            Z = self.drop(h)
            weights = self.proj(Z).squeeze()
            scores = self.sigmoid(weights)
            un_g, new_h, idx = top_k_graph(scores, g, h, self.k)
            new_edge_index = ((torch.tril(un_g, diagonal=-1) == 1).nonzero(as_tuple=False)).T
            graph.x = new_h
            graph.edge_index = new_edge_index
            data_list_out.append(graph)
        batch_out = pygb.Batch.from_data_list(data_list_out)
        return batch_out
