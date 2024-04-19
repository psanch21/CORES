from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from cores.core.values.constants import ActivationTypes
from cores.impl.mlp.mlp_torch import MLP


def create_tensor(x_num: int, x_dim: int, seed: int) -> torch.Tensor:
    np.random.seed(seed)
    torch.manual_seed(seed)

    x = torch.rand((x_num, x_dim))

    return x


def create_kwargs(seed):
    np.random.seed(seed)
    kwargs = {}
    kwargs["output_dim"] = random.choice([8, 16, 32, 64, 128])
    kwargs["layers_num"] = np.random.randint(1, 4)

    # Sample from ActivationTypes
    kwargs["activation"] = random.choice(list(ActivationTypes))
    kwargs["device"] = "cpu"
    kwargs["hidden_dim"] = np.random.randint(1, 64)
    kwargs["has_bn"] = random.choice([True, False])
    kwargs["has_ln"] = random.choice([True, False])
    kwargs["use_act_out"] = random.choice([True, False])
    kwargs["dropout"] = random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
    kwargs["drop_last"] = random.choice([True, False])

    return kwargs


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("n", [1, 32])
@pytest.mark.parametrize("x_dim", [1, 16])
@pytest.mark.parametrize("seed", list(range(150)))
def test_mlp_torch(device: str, n: int, x_dim: int, seed: int):
    x = create_tensor(n, x_dim, seed)

    kwargs = create_kwargs(seed)

    kwargs["input_dim"] = x_dim

    for key, value in kwargs.items():
        print(f"{key}: {value}")
    mlp = MLP(**kwargs)

    if kwargs["has_bn"] and n == 1:
        with pytest.raises(ValueError):
            out = mlp(x)
    else:
        out = mlp(x)
        assert out.shape == (n, kwargs["output_dim"])
        if not kwargs["use_act_out"] and not kwargs["drop_last"]:
            assert out.min() < 0.0
            assert out.max() > 0.0
