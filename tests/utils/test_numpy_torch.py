from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from cores.utils.numpy import NPUtils
from cores.utils.torch import TorchUtils


@pytest.mark.parametrize("q", [0.1, 0.2, 0.3, 0.4, 0.5])
def test_quantile(q: float):
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    quantile_np = np.quantile(scores, q, method="higher")
    quantile_torch = torch.quantile(torch.tensor(scores), q, interpolation="higher")

    assert np.allclose(
        quantile_np, quantile_torch.numpy()
    ), f"{quantile_np} != {quantile_torch.numpy()}"

    assert quantile_np.ndim == 0
    assert quantile_torch.ndim == 0


@pytest.mark.parametrize("seed", list(range(100)))
def test_sort_sum(seed: int):
    np.random.seed(seed)

    # Randomly sample the classes_num from 3 to 20
    classes_num = np.random.randint(3, 20)
    batch_size = 5

    logits = np.random.randn(batch_size, classes_num)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    assert np.allclose(probs.sum(axis=1), 1)
    assert probs.shape[-1] == classes_num

    I, ordered, cumsum = NPUtils.sort_sum(probs)
    I_2, ordered_2, cumsum_2 = TorchUtils.sort_sum(torch.tensor(probs))

    I_2 = I_2.cpu().numpy()
    ordered_2 = ordered_2.cpu().numpy()
    cumsum_2 = cumsum_2.cpu().numpy()

    assert np.allclose(I, I_2)
    assert np.allclose(ordered, ordered_2)
    assert np.allclose(cumsum, cumsum_2)

    assert I.shape == (batch_size, classes_num)
    assert ordered.shape == (batch_size, classes_num)
    assert cumsum.shape == (batch_size, classes_num)

    assert np.allclose(cumsum[:, -1], 1)


@pytest.mark.parametrize("is_balanced", [True, False])
@pytest.mark.parametrize("seed", list(range(100)))
def test_resample(is_balanced: bool, seed: int):
    np.random.seed(seed)

    # Randomly sample the classes_num from 3 to 20
    classes_num = np.random.randint(3, 20)
    batch_size = 20 * classes_num
    upsample = random.choice([True, False])

    if is_balanced:
        # Sample 20 samples from each class
        target = np.repeat(np.arange(classes_num), 20)
    else:
        target = np.random.randint(0, classes_num, size=batch_size)

    assert target.ndim == 1
    assert target.shape[0] == batch_size

    indeces = NPUtils.resample(target=target, upsample=upsample, replace=True, random_state=seed)

    y_resampled = target[indeces]

    if is_balanced:
        assert y_resampled.shape[0] == batch_size
    else:
        if upsample:
            assert y_resampled.shape[0] > batch_size
        else:
            assert y_resampled.shape[0] < batch_size

    assert y_resampled.ndim == 1
    assert indeces.ndim == 1

    # Count the number of elements per class
    count = np.bincount(y_resampled)

    # Assert all elements in count are equal
    assert np.all(count == count[0])
