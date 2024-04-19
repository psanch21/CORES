from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from cores.utils import NPUtils, TorchUtils
from cores.utils.conformal import ConformalAPSNP, ConformalAPSTorch


def generate_data(classes_num: int, batch_size: int, is_random: bool = False):
    y = np.random.randint(0, classes_num, (batch_size,))

    if is_random:
        logits = np.random.rand(batch_size, classes_num)
    else:
        # Convert y to one-hot
        y_onehot = np.zeros((batch_size, classes_num))
        y_onehot[np.arange(batch_size), y] = 1

        logits = y_onehot * 3 - 1.5 + np.random.rand(batch_size, classes_num) * 0.3

    u_values = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    if classes_num == 2:
        u_values = u_values[:, [1]]

    return u_values, y


@pytest.mark.parametrize("batch_size", [1000])
@pytest.mark.parametrize("is_random", [True, False])
@pytest.mark.parametrize("mode", ["torch", "numpy"])
@pytest.mark.parametrize("seed", list(range(10)))
def test_clf(batch_size: int, is_random: bool, mode: str, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    classes_num = random.choice([2, 3, 10])

    u_cal, y_cal = generate_data(classes_num, batch_size, is_random=is_random)

    u_val, y_val = generate_data(classes_num, batch_size, is_random=is_random)
    if mode == "torch":
        cmodel = ConformalAPSTorch(classes_num=classes_num, alpha=0.1)

        u_cal = torch.tensor(u_cal).to(torch.float32)
        y_cal = torch.tensor(y_cal)

        u_val = torch.tensor(u_val).to(torch.float32)
        y_val = torch.tensor(y_val)

        u_cal = TorchUtils.expand_binary_probs(u_cal)
        u_val = TorchUtils.expand_binary_probs(u_val)

        arange = torch.arange(batch_size)

        allclose_fn = torch.allclose
        ceil_fn = np.ceil
        all_fn = torch.all
        mean_fn = torch.mean
    else:
        cmodel = ConformalAPSNP(classes_num=classes_num, alpha=0.1)

        arange = np.arange(batch_size)

        u_cal = NPUtils.expand_binary_probs(u_cal)
        u_val = NPUtils.expand_binary_probs(u_val)

        allclose_fn = np.allclose
        ceil_fn = np.ceil
        all_fn = np.all
        mean_fn = np.mean

    scores = cmodel.score_fn(u_cal, y_cal)

    assert scores.ndim == 1
    assert scores.shape[0] == batch_size

    if not is_random:
        assert allclose_fn(scores, u_cal[arange, y_cal])

    cmodel.fit(u_cal, y_cal)

    y_set = cmodel(u_val)

    assert y_set.shape == (batch_size, classes_num)
    if not is_random:
        mean = ceil_fn(classes_num * 0.2)
        assert all_fn(y_set.sum(axis=1) <= mean)
    else:
        assert all_fn(y_set.sum(axis=1) > 1)

    if not is_random:
        coverages = cmodel.coverage_distribution(u_cal, y_cal, u_val, y_val, iterations=500)

        # Plot the histogram of coverages
        coverages_true = cmodel.sample_coverage_distribution(500)
        plt.figure()

        plt.hist(coverages, bins=20, color="blue", alpha=0.5)
        plt.hist(coverages_true, bins=20, color="red", alpha=0.5)

        # Save plot
        plt.savefig(os.path.join("tests", "images", f"coverage_{seed}_{int(is_random)}.png"))
        plt.close("all")
        success, p_value = cmodel.coverage_ks_test(coverages, significance_level=0.05)

        c_mean = mean_fn(coverages)

        assert c_mean > 0.0
