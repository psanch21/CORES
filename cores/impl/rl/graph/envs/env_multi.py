from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch_geometric.data as pygd

import cores.core.values.constants as Cte
from cores.impl.rl.graph.envs.env_base import GraphEnvBase
from cores.utils import PyGUtils


class GraphEnv(GraphEnvBase):
    def __init__(self, *args, **kwargs):
        super(GraphEnv, self).__init__(*args, **kwargs)

    def _remove(self, state: pygd.Data, idx_to_remove: List[int], relabel_nodes: bool) -> pygd.Data:
        if self.action_refers_to == Cte.ActionTypes.EDGE:
            data = PyGUtils.remove_edges_from_batch(
                edges_to_remove=idx_to_remove, batch=state.clone()
            )
            return data

        elif self.action_refers_to == Cte.ActionTypes.NODE:
            data = PyGUtils.remove_nodes_from_batch(
                batch=state.clone(), nodes_idx=idx_to_remove, relabel_nodes=relabel_nodes
            )

            return data
        else:
            raise NotImplementedError

    def _step(self, action: torch.Tensor, relabel_nodes: bool = True):
        # Execute one time step within the environment
        if not all([a in [0.0, 1.0] for a in action.unique()]):
            raise ValueError("Action must be a tensor of 0s and 1s")

        idx_to_remove = list(np.where(action.cpu() == 1)[0])

        num_idx_to_remove = int(action.sum().item())

        info = {"Elements to remove": 1, "Finish?": "No", "current_iter": self.current_iter}

        if num_idx_to_remove > 0:  # We need to remove some nodes
            data = self._remove(
                state=self.state, idx_to_remove=idx_to_remove, relabel_nodes=relabel_nodes
            )

            if self._is_empty(data):
                done = True
                info["Finish?"] = "Yes, result would have no nodes/edges"
                new_state = self.state

                reward = self.reward_fn.compute(state=self.state, action=action)
                penalty = self.penalty_size * torch.ones(1, device=self.device)
                reward -= penalty

                return new_state, reward, done, info
            else:
                self.state = data
                reward = self.reward_fn.compute(state=self.state, action=action)

        elif self._is_original(self.state):
            reward = self.reward_fn.compute(state=self.state, action=action)
            penalty = self.penalty_size * torch.ones(1, device=self.device)
            reward -= penalty
        else:
            reward = self.reward_fn.compute(state=self.state, action=action)

        intrinsic_reward = torch.zeros(1, device=self.device)

        self.current_reward = reward  # - self.init_reward

        if (num_idx_to_remove == 0) or self.current_iter == self.max_episode_length:
            done = True
            info["Finish?"] = "Yes, No elements to remove"
            return self.state, self.current_reward, done, info
        else:
            done = False
            return self.state, intrinsic_reward, done, info

    def get_action_distr_name(self) -> Cte.Distributions:
        return Cte.Distributions.BERNOULLI
