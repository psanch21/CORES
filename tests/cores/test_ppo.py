from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch_geometric.datasets.fake import FakeDataset
from torch_geometric.loader import DataLoader

import cores.core.values.constants as cte
from cores.impl.gnn.pyg.layers import GAT, GCN
from cores.impl.rl.graph.policy import GraphActorCritic
from cores.impl.rl.graph.ppo import GraphPPO


def create_data(seed):
    np.random.seed(seed)
    avg_num_nodes = random.choice([1, 8, 32])
    avg_degree = random.choice([1, 3])
    num_channels = random.choice([8, 16])
    edge_dim = random.choice([0, 4, 8])

    num_classes = random.choice([2, 10])
    dataset = FakeDataset(
        num_graphs=16,
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
        num_channels=num_channels,
        edge_dim=edge_dim,
        num_classes=num_classes,
        task="auto",
    )
    return dataset, num_classes


@pytest.mark.parametrize("seed", list(range(50)))
def test_ppo(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    dataset, classes_num = create_data(seed)

    data = dataset[0]

    loader = DataLoader(dataset, batch_size=4)

    x_dim = data.x.shape[1]

    GNN = random.choice([GCN, GAT])

    kwargs = GNN.random_kwargs(seed)
    kwargs["has_bn"] = False
    kwargs["input_dim"] = x_dim
    kwargs["pooling"] = None

    gnn = GNN(**kwargs)

    action_refers_to = random.choice(list(cte.ActionTypes))

    pool_1 = cte.PoolingTypes.MEAN
    pool_2 = cte.PoolingTypes.MAX
    pool_3 = [cte.PoolingTypes.MEAN, cte.PoolingTypes.ADD]

    pool_type = random.choice([pool_1, pool_2, pool_3])
    action_distr = random.choice([cte.Distributions.BERNOULLI, cte.Distributions.CONT_BERNOULLI])
    policy = GraphActorCritic(
        gnn=gnn,
        action_refers_to=action_refers_to,
        pool_type=pool_type,
        action_distr=action_distr,
        activation=cte.ActivationTypes.RELU,
    )

    eps_clip = np.random.rand()
    # Sample gamma from 0.9 to 0.999
    gamma = 0.9 + 0.099 * np.random.rand()

    # Sample coeff_mse from 0.1 to 10
    coeff_mse = 0.1 + 9.9 * np.random.rand()
    coeff_entropy = 0.001 + 0.099 * np.random.rand()

    ppo = GraphPPO(
        policy=policy,
        eps_clip=eps_clip,
        gamma=gamma,
        coeff_mse=coeff_mse,
        coeff_entropy=coeff_entropy,
    )

    batch = next(iter(loader))
    for data in batch.to_data_list():
        if data.num_edges == 0:
            return

    return_logprobs = random.choice([True, False])
    sample = random.choice([True, False])
    values = random.choice([True, False])

    act_dict = ppo.act(state=batch, return_logprobs=return_logprobs, sample=sample, values=values)

    if return_logprobs:
        assert "action_logprobs" in act_dict
        action_logprobs = act_dict["action_logprobs"]
        assert action_logprobs.ndim == 1
        assert action_logprobs.shape[0] == batch.num_graphs

    if sample:
        assert "action" in act_dict
        action_sample = act_dict["action"]
        assert action_sample.ndim == 1
        if action_refers_to == cte.ActionTypes.NODE:
            assert action_sample.shape[0] == batch.num_nodes
        elif action_refers_to == cte.ActionTypes.EDGE:
            assert action_sample.shape[0] == batch.num_edges
        else:
            raise ValueError(f"Unknown action_refers_to {action_refers_to}")
    else:
        assert "action" in act_dict
        action_mode = act_dict["action"]
        assert action_mode.ndim == 1
        if action_refers_to == cte.ActionTypes.NODE:
            assert action_mode.shape[0] == batch.num_nodes
        elif action_refers_to == cte.ActionTypes.EDGE:
            assert action_mode.shape[0] == batch.num_edges
        else:
            raise ValueError(f"Unknown action_refers_to {action_refers_to}")

    if values:
        assert "state_values" in act_dict
        state_values = act_dict["state_values"]
        assert state_values.ndim == 2
        assert state_values.shape[1] == 1, f"{state_values.shape}"
        assert state_values.shape[0] == batch.num_graphs
