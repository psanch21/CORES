from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch_geometric.datasets.fake import FakeDataset
from torch_geometric.loader import DataLoader

import cores.impl.rl.graph.rewards as grew
from cores.core.values.constants import PoolingTypes
from cores.impl.gnn.pyg.layers import GAT, GCN


def create_data(seed: int, num_graphs: int = 8):
    np.random.seed(seed)
    avg_num_nodes = random.choice([1, 8, 32, 1024])
    avg_degree = random.choice([1, 20])
    num_channels = random.choice([8, 16])
    edge_dim = random.choice([0, 4, 8])

    num_classes = random.choice([2, 10])
    dataset = FakeDataset(
        num_graphs=num_graphs,
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
        num_channels=num_channels,
        edge_dim=edge_dim,
        num_classes=num_classes,
        task="auto",
    )
    return dataset, num_classes


@pytest.mark.parametrize("mode", ["data", "batch"])
@pytest.mark.parametrize("seed", list(range(50)))
def test_reward_conformal(mode: str, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    dataset, classes_num = create_data(seed, num_graphs=64)
    loader = DataLoader(dataset, batch_size=4)

    if mode == "data":
        data = dataset[0]

    elif mode == "batch":
        data = next(iter(loader))
        assert data.batch is not None

    x_dim = data.x.shape[1]

    GNN = random.choice([GCN, GAT])

    kwargs = GNN.random_kwargs(seed)
    kwargs["has_bn"] = False
    kwargs["input_dim"] = x_dim
    kwargs["pooling"] = [PoolingTypes.MEAN]

    if classes_num == 2:
        kwargs["output_dim"] = 1
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        kwargs["output_dim"] = classes_num
        loss_fn = torch.nn.CrossEntropyLoss()

    graph_clf = GNN(**kwargs)
    kwargs = grew.RewardConformal.random_kwargs(seed)

    assert data.y.ndim == 1

    if mode == "data":
        assert data.y.numel() == 1
    else:
        assert data.y.numel() == data.batch.unique().numel()

    reward = grew.RewardConformal(
        graph_clf=graph_clf,
        classes_num=classes_num,
        loss_fn=loss_fn,
        device="cpu",
        loader_calib=loader,
        **kwargs,
    )

    reward.fit_conformal_from_loader()

    if data.num_edges == 0:
        return

    if mode == "batch":
        # Assert raise NotImplementedError
        with pytest.raises(NotImplementedError):
            reward.set_state_0(state_0=data.clone())
    else:
        reward.set_state_0(state_0=data.clone())

        logits = graph_clf(data.clone())

        assert logits.ndim == 2
        assert logits.shape[0] == 1
        value = reward.compute(state=data)

        assert isinstance(value, torch.FloatTensor)

        assert value.ndim == 1
        assert value.shape[0] == 1
        assert value.numel() == 1
