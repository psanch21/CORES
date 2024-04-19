from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from cores.utils import BinaryEntropy, CategoricalEntropy


@pytest.mark.parametrize("seed", list(range(50)))
def test_entropy_binary(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    entropy = BinaryEntropy()

    logits = torch.randn(10, 1)

    if np.random.rand() > 0.5:
        logits = logits.flatten()

    value = entropy(logits)

    value_max = entropy.entropy_max(logits)

    assert value.shape == (10,)
    assert value.ndim == 1
    assert value_max.numel() == 1

    assert (value <= value_max).all()


@pytest.mark.parametrize("seed", list(range(50)))
def test_entropy_cat(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    entropy = CategoricalEntropy()

    classes_num = np.random.randint(2, 20)

    batch_size = random.choice([1, 5, 10, 20])

    logits = torch.randn(batch_size, classes_num)

    if batch_size == 1 and np.random.rand() > 0.5:
        logits = logits.flatten()

    value = entropy(logits)

    value_max = entropy.entropy_max(logits)

    assert value.shape == (batch_size,)
    assert value.ndim == 1
    assert value_max.numel() == 1

    assert (value <= value_max).all()
