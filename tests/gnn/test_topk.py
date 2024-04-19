from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch_geometric.datasets.fake import FakeDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool.topk_pool import TopKPooling

import cores.impl.gnn.pyg.layers as ggnn
from cores.impl.gnn.pyg.gnn_base import BaseGNN
from cores.impl.gnn.pyg.gnn_top_k import TopKGNN


def create_data(seed):
    np.random.seed(seed)
    avg_num_nodes = random.choice([8, 32])
    avg_degree = random.choice([1, 5])
    num_channels = random.choice([8, 16])
    edge_dim = random.choice([8, 16])

    dataset = FakeDataset(
        num_graphs=32,
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
        num_channels=num_channels,
        edge_dim=edge_dim,
        num_classes=10,
        task="auto",
    )
    return dataset


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("GNN", [ggnn.GCN, ggnn.GAT, ggnn.GIN, ggnn.GINE])
@pytest.mark.parametrize("seed", list(range(20)))
def test_topk(device: str, GNN: BaseGNN, seed: int):
    # Sample a random data from the dataset
    np.random.seed(seed)
    random.seed(seed)
    dataset = create_data(seed)

    loader = DataLoader(dataset, batch_size=32)

    batch = next(iter(loader))

    x_dim = batch.x.shape[1]

    kwargs = GNN.random_kwargs(seed)

    kwargs["input_dim"] = x_dim
    kwargs["pooling"] = None

    if GNN in [ggnn.GINE, ggnn.GAT]:
        kwargs["edge_dim"] = batch.edge_attr.shape[1]

    # for key, value in kwargs.items():
    #     print(f"{key}: {value}")

    gnn = GNN(**kwargs)

    use_gnn = np.random.rand() < 0.5

    kwargs_topk = TopKGNN.random_kwargs(seed)

    if use_gnn:
        top_k = TopKGNN(in_channels=gnn.output_dim, gnn=gnn, **kwargs_topk)
    else:
        top_k = TopKGNN(in_channels=x_dim, gnn=None, **kwargs_topk)

    batch_2 = top_k(batch=batch)

    assert batch_2.x.shape[0] <= batch.x.shape[0]
    assert batch_2.x.shape[1] == batch.x.shape[1]

    assert batch_2.edge_index.shape[1] <= batch.edge_index.shape[1]


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("seed", list(range(150)))
def test_module(device: str, seed: int):
    # Sample a random data from the dataset
    np.random.seed(seed)
    dataset = create_data(seed)

    loader = DataLoader(dataset, batch_size=32)

    batch = next(iter(loader))

    in_channels = batch.x.shape[1]

    kwargs_topk = TopKGNN.random_kwargs(seed)

    pool = TopKPooling(in_channels=in_channels, nonlinearity=torch.tanh, **kwargs_topk)

    logits = torch.randn(batch.x.shape[0], 1)

    edge_index_0 = batch.edge_index.clone()

    x, edge_index, edge_attr, batch_, perm, score = pool(
        x=batch.x,
        edge_index=batch.edge_index,
        batch=batch.batch,
        edge_attr=batch.edge_attr,
        attn=logits,
    )

    assert edge_index.shape[1] <= edge_index_0.shape[1]

    if edge_index.numel() > 0:
        assert edge_index.max() < batch.x.shape[0]
