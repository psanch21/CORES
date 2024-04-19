from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch
from torch_geometric.datasets.fake import FakeDataset
from torch_geometric.loader import DataLoader

from cores.impl.gnn.pyg.g_pool import GPool


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
@pytest.mark.parametrize("seed", list(range(20)))
def test_gpool(device: str, seed: int):
    # Sample a random data from the dataset
    np.random.seed(seed)
    random.seed(seed)
    dataset = create_data(seed)

    loader = DataLoader(dataset, batch_size=32)

    batch = next(iter(loader))

    x_dim = batch.x.shape[1]

    kwargs_gpool = GPool.random_kwargs(seed)
    kwargs_gpool["device"] = device

    g_pool = GPool(in_channels=x_dim, **kwargs_gpool)

    batch_2 = g_pool(batch=batch)

    assert batch_2.x.shape[0] <= batch.x.shape[0]
    assert batch_2.x.shape[1] == batch.x.shape[1]

    assert batch_2.edge_index.shape[1] <= batch.edge_index.shape[1]
