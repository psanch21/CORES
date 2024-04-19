from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch_geometric.data as pygd

import cores.core.values.constants as cte
from cores.impl.rl.graph.rewards.reward_base import Reward
from cores.utils import TorchUtils
from cores.utils.conformal import ConformalAPSTorch


class RewardConformal(Reward):
    @staticmethod
    def random_kwargs(seed: int) -> Dict[str, Any]:
        kwargs = {}
        kwargs["lambda_1"] = np.random.rand()
        kwargs["alpha"] = np.random.rand() * 0.1 + 0.05
        kwargs["desired_ratio"] = np.random.rand()

        kwargs["action_refers_to"] = random.choice(list(cte.ActionTypes))

        return kwargs

    def __init__(
        self,
        *args,
        lambda_1: float = 0.5,
        desired_ratio: float = 0.9,
        action_refers_to: cte.ActionTypes = None,
        alpha: float = 0.1,
        k: float = 1.0,
        allow_worsen: bool = False,
        loader_calib: Optional[pygd.DataLoader] = None,
        **kwargs,
    ):
        assert desired_ratio > 0.0
        assert desired_ratio < 1.0
        assert isinstance(action_refers_to, cte.ActionTypes)

        for lamba_ in [lambda_1]:
            assert lamba_ >= 0.0
            assert lamba_ <= 1.0

        self.lambda_1 = lambda_1

        self.k = k
        self.alpha = alpha
        self.allow_worsen = allow_worsen

        self.loader_calib = loader_calib

        self.desired_ratio = desired_ratio
        self.action_refers_to = action_refers_to

        self.exp_cte = np.log(1.0 - 0.95) / np.log(self.desired_ratio)

        self.cmodel = None

        super(RewardConformal, self).__init__(*args, **kwargs)

    def fit_conformal(self, logits: torch.FloatTensor, target: torch.LongTensor):
        cmodel = ConformalAPSTorch(classes_num=self.classes_num, alpha=self.alpha)

        probs = self.logits_to_soft_pred(logits=logits)

        probs_exp = TorchUtils.expand_binary_probs(probs)

        if target.ndim == 2:
            assert target.shape[1] == 1
        target = target.flatten()

        cmodel.fit(probs_exp, target)

        self.cmodel = cmodel

    def fit_conformal_from_loader(
        self, loader: Optional[pygd.DataLoader] = None, batch_norm_fn: Optional[Callable] = None
    ):
        logits, target = [], []

        if loader is None:
            loader = self.loader_calib

        for batch in loader:
            if batch_norm_fn is not None:
                batch_ = batch_norm_fn(batch, policy_kwargs={"sample": True, "num_samples": 1}).to(
                    self.device
                )
            else:
                batch_ = batch.clone().to(self.device)
            logits.append(self.graph_clf(batch_))
            target.append(self.batch_to_target(batch_))

        logits = torch.cat(logits, dim=0)
        target = torch.cat(target, dim=0)

        self.fit_conformal(logits=logits, target=target)

    def compute_reward_sparsity(self, state: pygd.Batch):
        if self.action_refers_to == cte.ActionTypes.NODE:
            ratio = state.x.shape[0] / self.state_0.x.shape[0]
        elif self.action_refers_to == cte.ActionTypes.EDGE:
            ratio = state.edge_index.shape[1] / self.state_0.edge_index.shape[1]
        else:
            raise NotImplementedError(f"Action refers to {self.action_refers_to} not implemented")

        desired_ratio_cte = 1.0 - (ratio**self.exp_cte)
        reward_sparsity = desired_ratio_cte**2
        return reward_sparsity

    def _compute(self, logits: torch.FloatTensor, state: pygd.Batch, **kwargs):
        assert logits.ndim == 2
        assert logits.shape[0] == 1
        if self.cmodel is None:
            return torch.zeros(1).unsqueeze(dim=-1).to(logits.device)

        num_samples = self._num_samples(logits)

        probs = self.logits_to_soft_pred(logits)

        probs_0 = self.logits_to_soft_pred(self.logits_0)

        probs_exp = TorchUtils.expand_binary_probs(probs)
        probs_0_exp = TorchUtils.expand_binary_probs(probs_0)
        y_set = self.cmodel.compute_conformal_set(probs_exp)

        assert y_set.ndim == 2
        assert y_set.shape[0] == 1
        assert self.target.numel() == 1

        reward_sparsity = self.compute_reward_sparsity(state)

        reward_sparsity = reward_sparsity * torch.ones(num_samples).to(logits.device)

        target = self.target.long()

        is_correct = y_set[0, target].item() == 1
        y_set_size = y_set.sum()

        reward_performance = probs_exp[0, target]
        reward_performance_0 = probs_0_exp[0, target]

        if not self.allow_worsen and reward_performance < reward_performance_0 * 0.99:
            return -torch.ones(1).to(logits.device).unsqueeze(0)

        if is_correct:
            if y_set_size == 1:
                reward = self.lambda_1 * reward_performance + (1 - self.lambda_1) * reward_sparsity
                return self.k * reward.unsqueeze(0)

            else:
                reward = reward_performance / y_set_size
                return reward.unsqueeze(0)
        else:
            reward = -reward_sparsity
            return reward.unsqueeze(0)

    def __str__(self):
        my_str = super(RewardConformal, self).__str__()
        my_str += f"\tlambda_1={self.lambda_1}\n"

        my_str += f"\tdesired_ratio={self.desired_ratio}\n"
        my_str += f"\alpha={self.alpha}\n"

        return my_str
