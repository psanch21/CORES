from __future__ import annotations

import random
from typing import List

import numpy as np
import pytest
import torch

from cores.impl.metrics import BinaryCLFMetricsTorch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_metrics_list(seed: int) -> List[str]:
    metrics = ["accuracy", "f1", "precision", "recall"]
    num_metrics = np.random.randint(1, len(metrics) + 1)
    metrics_list = random.sample(metrics, num_metrics)
    return metrics_list


def create_logits(x_num: int, x_dim: int, seed: int = None) -> torch.Tensor:
    if seed is not None:
        set_seed(seed)

    x = torch.randn((x_num, x_dim))

    return x


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_size", [1, 2, 16, 32])
@pytest.mark.parametrize("num_batches", [1, 2, 10])
@pytest.mark.parametrize("seed", list(range(20)))
def test_clf_binary_perfect(device: str, batch_size: int, num_batches: int, seed: int):
    set_seed(seed)

    print(f"batch_size: {batch_size}")
    print(f"num_batches: {num_batches}")

    metrics_list = create_metrics_list(seed)

    metrics = BinaryCLFMetricsTorch(classes_num=2, full_names=metrics_list)
    # for m in metrics_list:
    #     metrics.add(m)

    for i in range(num_batches):
        logits = create_logits(batch_size, 1)

        target = torch.zeros((batch_size, 1), dtype=torch.long)
        target[logits > 0.0] = 1

        if target.sum() == 0:
            target[0] = 1
            logits[0] = 1234.024

        print(f"target: {target}")
        assert logits.shape == (batch_size, 1)
        assert target.shape == (batch_size, 1)
        for m in metrics_list:
            metrics.update(m, logits=logits, target=target)
    for m in metrics_list:
        metrics_dict = metrics.compute(m)
        assert isinstance(metrics_dict, dict)
        assert len(metrics_dict) == 1
        for key, value_torch in metrics_dict.items():
            value = value_torch.item()
        assert isinstance(value, float)
        assert value == 1.0, f"metric: {m} - {value}"


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("batch_size", [2, 16, 32])
@pytest.mark.parametrize("num_batches", [1, 2, 10])
@pytest.mark.parametrize("seed", list(range(10)))
def test_clf_binary_random(device: str, batch_size: int, num_batches: int, seed: int):
    metrics = BinaryCLFMetricsTorch(classes_num=2)

    metrics_list = ["accuracy", "f1", "precision", "recall"]
    for m in metrics_list:
        metrics.add(m)

    for i in range(num_batches):
        logits = create_logits(batch_size, 1, seed)

        target = torch.randint(0, 2, (batch_size, 1))

        assert logits.shape == (batch_size, 1)
        assert target.shape == (batch_size, 1)
        for m in metrics_list:
            metrics.update(m, logits=logits, target=target)
    for m in metrics_list:
        metrics_dict = metrics.compute(m)
        assert isinstance(metrics_dict, dict)
        for key, value_torch in metrics_dict.items():
            value = value_torch.item()
            assert isinstance(value, float)
            assert value >= 0.0
            assert value <= 1.0
