from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch_geometric.datasets.fake import FakeDataset
from torch_geometric.loader import DataLoader

import cores.core.values.constants as cte
from cores.impl.gnn.pyg.layers import GAT, GCN
from cores.impl.rl.graph.envs import GraphEnv, GraphEnvOne
from cores.impl.rl.graph.rewards import RewardConformal
from cores.utils.entropy import BinaryEntropy, CategoricalEntropy


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
def test_env_one(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    dataset, classes_num = create_data(seed)

    data = dataset[0]

    loader = DataLoader(dataset, batch_size=1)

    x_dim = data.x.shape[1]

    GNN = random.choice([GCN, GAT])

    kwargs = GNN.random_kwargs(seed)
    kwargs["has_bn"] = False
    kwargs["input_dim"] = x_dim
    kwargs["pooling"] = [cte.PoolingTypes.MEAN]

    if classes_num == 2:
        kwargs["output_dim"] = 1
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        kwargs["output_dim"] = classes_num
        loss_fn = torch.nn.CrossEntropyLoss()

    graph_clf = GNN(**kwargs)
    kwargs = RewardConformal.random_kwargs(seed)

    assert data.y.ndim == 1

    reward = RewardConformal(
        graph_clf=graph_clf,
        classes_num=classes_num,
        loss_fn=loss_fn,
        device="cpu",
        **kwargs,
    )

    action_refers_to = random.choice(list(cte.ActionTypes))
    penalty_size = random.choice([0.0, 1.0])
    max_episode_length = random.choice([1, 5, 10])

    use_intrinsic_reward = False

    env = GraphEnvOne(
        loader=loader,
        graph_clf=graph_clf,
        reward_fn=reward,
        action_refers_to=action_refers_to,
        penalty_size=penalty_size,
        max_episode_length=max_episode_length,
        use_intrinsic_reward=use_intrinsic_reward,
    )

    graph = next(iter(env.loader))

    if graph.num_edges == 0:
        # raise ValueError("state_0 has not edges")
        with pytest.raises(ValueError):
            state = env.reset(graph=graph)
    else:
        state = env.reset(graph=graph)
        done = False

        while not done:
            if action_refers_to == cte.ActionTypes.EDGE:
                action_dim = state.num_edges
            elif action_refers_to == cte.ActionTypes.NODE:
                action_dim = state.num_nodes

            action = torch.rand((action_dim,))
            state, reward, done, info = env.step(action)


@pytest.mark.parametrize("seed", list(range(50)))
def test_env_multi(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    dataset, classes_num = create_data(seed)

    data = dataset[0]

    loader = DataLoader(dataset, batch_size=1)

    x_dim = data.x.shape[1]

    GNN = random.choice([GCN, GAT])

    kwargs = GNN.random_kwargs(seed)
    kwargs["has_bn"] = False
    kwargs["input_dim"] = x_dim
    kwargs["pooling"] = [cte.PoolingTypes.MEAN]

    if classes_num == 2:
        kwargs["output_dim"] = 1
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        kwargs["output_dim"] = classes_num
        loss_fn = torch.nn.CrossEntropyLoss()

    graph_clf = GNN(**kwargs)
    kwargs = RewardConformal.random_kwargs(seed)

    assert data.y.ndim == 1

    reward = RewardConformal(
        graph_clf=graph_clf,
        classes_num=classes_num,
        loss_fn=loss_fn,
        device="cpu",
        **kwargs,
    )

    action_refers_to = random.choice(list(cte.ActionTypes))
    penalty_size = random.choice([0.0, 1.0])
    max_episode_length = random.choice([1, 5, 10])

    use_intrinsic_reward = False

    env = GraphEnv(
        loader=loader,
        graph_clf=graph_clf,
        reward_fn=reward,
        action_refers_to=action_refers_to,
        penalty_size=penalty_size,
        max_episode_length=max_episode_length,
        use_intrinsic_reward=use_intrinsic_reward,
    )

    graph = next(iter(env.loader))

    if graph.num_edges == 0:
        # raise ValueError("state_0 has not edges")
        with pytest.raises(ValueError):
            state = env.reset(graph=graph)
    else:
        state = env.reset(graph=graph)
        done = False

        while not done:
            if action_refers_to == cte.ActionTypes.EDGE:
                action_dim = state.num_edges
            elif action_refers_to == cte.ActionTypes.NODE:
                action_dim = state.num_nodes

            # Sample action_dim bernoullies

            action_probs = torch.rand((action_dim,))
            action = (action_probs > 0.5).float()
            if np.random.rand() < 0.8:
                state, reward, done, info = env.step(action)
            else:
                with pytest.raises(ValueError):
                    state, reward, done, info = env.step(action_probs)
                state, reward, done, info = env.step(action)
