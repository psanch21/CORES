from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.datasets.fake import FakeDataset
from torch_geometric.loader import DataLoader

import cores.core.values.constants as cte
from cores.impl.gnn.pyg.layers import GAT, GCN
from cores.impl.rl.graph.policy import GraphActorCritic
from cores.impl.rl.graph.rewards import RewardConformal


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
def test_graph_actor_critic(seed: int):
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
    kwargs = RewardConformal.random_kwargs(seed)

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

    params_gnn_0 = list(gnn.parameters())[0].clone()
    params_actor_0 = list(policy.actor.parameters())[0].clone()
    params_critic_0 = list(policy.critic.parameters())[0].clone()

    init_fn_name = random.choice(["normal", "none"])

    if init_fn_name == "normal":

        def init_fn(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.5)
                if isinstance(module.bias, torch.Tensor):
                    nn.init.zeros_(module.bias)

    else:

        def init_fn(x):
            return

    policy.apply(init_fn)

    # gnn.apply(init_fn)

    params_gnn_1 = list(gnn.parameters())[0]
    params_actor_1 = list(policy.actor.parameters())[0]
    params_critic_1 = list(policy.critic.parameters())[0]

    if init_fn_name == "normal":
        assert not torch.allclose(params_gnn_0, params_gnn_1)
        assert not torch.allclose(params_actor_0, params_actor_1)
        assert not torch.allclose(params_critic_0, params_critic_1)

    else:
        assert torch.allclose(params_gnn_0, params_gnn_1)
        assert torch.allclose(params_actor_0, params_actor_1)
        assert torch.allclose(params_critic_0, params_critic_1)

    batch = next(iter(loader))
    for data in batch.to_data_list():
        if data.num_edges == 0:
            return
    x = batch.x.clone()
    actor_logits = policy.feature_extractor_actor(batch)

    assert torch.allclose(batch.x, x)

    assert actor_logits.shape[0] == batch.num_nodes
    action_logits = policy.compute_action_logits(batch)

    assert torch.allclose(batch.x, x)

    if action_refers_to == cte.ActionTypes.NODE:
        assert action_logits.shape[0] == batch.num_nodes
    elif action_refers_to == cte.ActionTypes.EDGE:
        assert action_logits.shape[0] == batch.num_edges
    else:
        raise ValueError(f"Unknown action_refers_to {action_refers_to}")

    critic_logits = policy.feature_extractor_critic(batch)

    assert torch.allclose(batch.x, x)
    assert critic_logits.shape[0] == batch.num_nodes

    state_values = policy.get_state_values(batch)

    assert state_values.ndim == 2
    assert state_values.shape[1] == 1, f"{state_values.shape}"
    assert state_values.shape[0] == batch.num_graphs

    action = policy.compute(state=batch, return_list=["action_sample"])["action_sample"]

    assert action.ndim == 1
    if action_refers_to == cte.ActionTypes.NODE:
        assert action.shape[0] == batch.num_nodes
    elif action_refers_to == cte.ActionTypes.EDGE:
        assert action.shape[0] == batch.num_edges
    else:
        raise ValueError(f"Unknown action_refers_to {action_refers_to}")

    action_logprobs, state_values, dist_entropy = policy.evaluate(state=batch, action=action)

    assert action_logprobs.ndim == 1

    assert state_values.ndim == 2
    assert state_values.shape[1] == 1, f"{state_values.shape}"
    assert state_values.shape[0] == batch.num_graphs

    assert dist_entropy.ndim == 1
