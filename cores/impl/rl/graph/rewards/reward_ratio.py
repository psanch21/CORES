from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch

import cores.core.values.constants as cte
from cores.impl.rl.graph.rewards.reward_base import Reward


class RewardRatio(Reward):
    @staticmethod
    def random_kwargs(seed: int) -> Dict[str, Any]:
        kwargs = {}
        kwargs["lambda_1"] = np.random.rand()
        kwargs["lambda_2"] = np.random.rand()
        kwargs["lambda_3"] = np.random.rand()

        kwargs["k_1"] = np.random.rand() * 10
        kwargs["k_2"] = np.random.rand() * 10
        kwargs["k_3"] = np.random.rand() * 10
        kwargs["action_refers_to"] = random.choice(list(cte.ActionTypes))

        return kwargs

    def __init__(
        self,
        *args,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float,
        k_1: float,
        k_2: float,
        k_3: float,
        desired_ratio: float,
        action_refers_to: cte.ActionTypes = None,
        **kwargs,
    ):
        assert desired_ratio > 0.0
        assert desired_ratio < 1.0
        assert isinstance(action_refers_to, cte.ActionTypes)

        for lamba_ in [lambda_1, lambda_2, lambda_3]:
            assert lamba_ >= 0.0
            assert lamba_ <= 1.0
        for k in [k_1, k_2, k_3]:
            assert k >= 0.0

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3

        self.desired_ratio = desired_ratio
        self.action_refers_to = action_refers_to

        super(RewardRatio, self).__init__(*args, **kwargs)

    def _reward_ratio(self, logits, r_perf, r_spar):
        """
        Positive reward:  cte*desired_ratio_cte
        Negative reward:   -cte
        Neutral reward: 0.
        Args:
            logits: Logits of the clf with the sub-graph

        Returns:
            reward scalar
        """

        # pb_io.print_debug_tensor(logits, "logits")

        pred_hard = self.logits_to_hard_pred(logits)
        # pb_io.print_debug_tensor(pred_hard, "pred_hard")
        # pb_io.print_debug_tensor(self.target, "target")

        k_1 = self.k_1 * torch.ones_like(pred_hard)
        k_2 = self.k_2 * torch.ones_like(pred_hard)
        k_3 = self.k_3 * torch.ones_like(pred_hard)

        lambda_1 = self.lambda_1 * torch.ones_like(pred_hard)
        lambda_2 = self.lambda_2 * torch.ones_like(pred_hard)
        lambda_3 = self.lambda_3 * torch.ones_like(pred_hard)

        sub_graph_is_correct = self._subgraph_is_correct(pred_hard)

        if self.full_graph_is_correct():  # Clf is correct!
            # print(f"FULL IS CORRECT")
            if sub_graph_is_correct:  # Sub-graph is correct!
                # print(f"SPARSE IS CORRECT")
                reward = self._reward_1(lambda_1, k_1, r_perf, r_spar)
                assert reward >= 0, f"reward: {reward} | {k_1} {r_spar}"
                return reward
            else:  # Sub-graph is wrong!
                # print(f"SPARSE IS WRONG")
                reward = -self._reward_2(lambda_2, k_2, r_perf, r_spar)
                # assert reward <= 0, f"reward: {reward}"
                return reward

        else:  # Clf is wrong!
            # print(f"FULL IS WRONG")
            if sub_graph_is_correct:  # Sub-graph is correct!
                # print(f"SPARSE IS CORRECT")
                reward = self._reward_3(lambda_3, k_3, r_perf, r_spar)
                assert reward >= 0, f"reward: {reward}"
                return reward
            else:
                # print(f"SPARSE IS WRONG")
                return 0.0 * torch.ones_like(pred_hard)

    def _reward_1(self, lambda_1, k_1, r_perf, r_spar):
        raise NotImplementedError

    def _reward_2(self, lambda_2, k_2, r_perf, r_spar):
        raise NotImplementedError

    def _reward_3(self, lambda_3, k_3, r_perf, r_spar):
        raise NotImplementedError

    def __str__(self):
        my_str = super(RewardRatio, self).__str__()
        my_str += f"\tlambda_1={self.lambda_1}\n"
        my_str += f"\tlambda_2={self.lambda_2}\n"
        my_str += f"\tlambda_3={self.lambda_3}\n"

        my_str += f"\tk_1={self.k_1}\n"
        my_str += f"\tk_2={self.k_2}\n"
        my_str += f"\tk_3={self.k_3}\n"

        my_str += f"\tdesired_ratio={self.desired_ratio}\n"

        return my_str
