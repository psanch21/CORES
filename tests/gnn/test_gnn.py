from __future__ import annotations

import os
import random

import numpy as np
import pytest
from torch_geometric.datasets.fake import FakeDataset

import cores.impl.gnn.pyg.layers as ggnn
from cores.impl.gnn.pyg.gnn_base import BaseGNN
from cores.impl.logger import WandBLogger


def create_data(seed):
    np.random.seed(seed)
    avg_num_nodes = random.choice([1, 8, 32, 1024])
    avg_degree = random.choice([0, 1, 20])
    num_channels = random.choice([8, 16])
    edge_dim = random.choice([8, 16])

    dataset = FakeDataset(
        num_graphs=1,
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
        num_channels=num_channels,
        edge_dim=edge_dim,
        num_classes=10,
        task="auto",
    )
    return dataset[0]


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("GNN", [ggnn.GCN, ggnn.GAT, ggnn.GIN, ggnn.GINE])
@pytest.mark.parametrize("seed", list(range(150)))
def test_gnn(device: str, GNN: BaseGNN, seed: int):
    # Sample a random data from the dataset
    np.random.seed(seed)
    data = create_data(seed)
    x_dim = data.x.shape[1]
    nodes_num = data.x.shape[0]

    kwargs = GNN.random_kwargs(seed)

    kwargs["input_dim"] = x_dim

    if GNN in [ggnn.GINE, ggnn.GAT]:
        kwargs["edge_dim"] = data.edge_attr.shape[1]

    for key, value in kwargs.items():
        print(f"{key}: {value}")

    gnn = GNN(**kwargs)

    if kwargs["has_bn"] and nodes_num == 1:
        with pytest.raises(ValueError):
            logits = gnn(batch=data)
    else:
        logits = gnn(batch=data)

        if kwargs["pooling"] is None:
            assert logits.shape == (nodes_num, kwargs["output_dim"])
        else:
            assert logits.shape == (1, kwargs["output_dim"])


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("GNN", [ggnn.GCN])
@pytest.mark.parametrize("seed", [0])
def test_save(device: str, GNN: BaseGNN, seed: int):
    # Sample a random data from the dataset
    np.random.seed(seed)
    data = create_data(seed)
    x_dim = data.x.shape[1]

    kwargs = GNN.random_kwargs(seed)

    kwargs["input_dim"] = x_dim

    for key, value in kwargs.items():
        print(f"{key}: {value}")

    gnn = GNN(**kwargs)

    folder = os.path.join("tests", "gnn", "models", f"{GNN.__name__}_{seed}")
    if not os.path.exists(folder):
        os.makedirs(folder)

    logger = WandBLogger(config=kwargs, project="test", dir=folder)

    logger.save_model(file_name=f"{GNN.__name__}_{seed}.pth", model=gnn)
