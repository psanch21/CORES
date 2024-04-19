from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch_geometric.datasets.fake import FakeDataset
from torch_geometric.loader import DataLoader

from cores.utils import PyGUtils


def create_data(seed):
    np.random.seed(seed)
    avg_num_nodes = random.choice([1, 8, 32, 1024])
    avg_degree = random.choice([0, 1, 20])
    num_channels = random.choice([8, 16])
    edge_dim = random.choice([0, 4, 8])

    dataset = FakeDataset(
        num_graphs=8,
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
        num_channels=num_channels,
        edge_dim=edge_dim,
        num_classes=10,
        task="auto",
    )
    return dataset


@pytest.mark.parametrize("mode", ["data", "batch"])
@pytest.mark.parametrize("seed", list(range(100)))
def test_remove_edges(mode: str, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    dataset = create_data(seed)

    if mode == "data":
        data = dataset[0]

    elif mode == "batch":
        loader = DataLoader(dataset, batch_size=4)
        data = next(iter(loader))
        assert data.batch is not None

    num_edges = data.edge_index.shape[1]

    # Sample the size of the edges to remove
    size = random.choice([0, num_edges // 4, num_edges // 2, int(num_edges * 0.9), num_edges])

    num_edges_to_keep = num_edges - size

    edges_to_remove = list(np.random.choice(num_edges, size=size, replace=False))

    # sort the edges to remove
    edges_to_remove = sorted(edges_to_remove)

    assert len(edges_to_remove) == size

    has_edge_attr = data.edge_attr is not None

    if has_edge_attr:
        edge_dim = data.edge_attr.shape[1]

        assert edge_dim > 0

    edge_index, edge_attr = PyGUtils.remove_edges(
        edges_to_remove=edges_to_remove,
        num_edges=num_edges,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )

    assert edge_index.shape[1] == num_edges_to_keep

    if has_edge_attr:
        assert edge_attr.shape[0] == num_edges_to_keep

    batch = PyGUtils.remove_edges_from_batch(batch=data, edges_to_remove=edges_to_remove)

    assert batch.edge_index.shape[1] == num_edges_to_keep
    if has_edge_attr:
        assert batch.edge_attr.shape[0] == num_edges_to_keep


@pytest.mark.parametrize("mode", ["data", "batch"])
@pytest.mark.parametrize("seed", list(range(100)))
def test_remove_nodes(mode: str, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    dataset = create_data(seed)

    if mode == "data":
        data = dataset[0]

    elif mode == "batch":
        loader = DataLoader(dataset, batch_size=4)
        data = next(iter(loader))
        assert data.batch is not None

    num_edges = data.edge_index.shape[1]

    # Sample the size of the edges to remove
    size = random.choice([0, num_edges // 4, num_edges // 2, int(num_edges * 0.9), num_edges])

    num_edges_to_keep = num_edges - size

    edges_to_remove = list(np.random.choice(num_edges, size=size, replace=False))

    # sort the edges to remove
    edges_to_remove = sorted(edges_to_remove)

    assert len(edges_to_remove) == size

    has_edge_attr = data.edge_attr is not None

    if has_edge_attr:
        edge_dim = data.edge_attr.shape[1]

        assert edge_dim > 0

    edge_index, edge_attr = PyGUtils.remove_edges(
        edges_to_remove=edges_to_remove,
        num_edges=num_edges,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )

    assert edge_index.shape[1] == num_edges_to_keep

    if has_edge_attr:
        assert edge_attr.shape[0] == num_edges_to_keep

    batch = PyGUtils.remove_edges_from_batch(batch=data, edges_to_remove=edges_to_remove)

    assert batch.edge_index.shape[1] == num_edges_to_keep
    if has_edge_attr:
        assert batch.edge_attr.shape[0] == num_edges_to_keep
