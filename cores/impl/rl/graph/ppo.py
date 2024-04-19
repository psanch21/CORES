from __future__ import annotations

import copy
import logging
import random
from typing import Dict

import torch
import torch.nn as nn
import torch_geometric.data as pygd

from cores.impl.rl.buffers import RolloutBuffer
from cores.impl.rl.graph.envs.env_base import GraphEnvBase
from cores.impl.rl.graph.policy import GraphActorCritic


class GraphPPO(nn.Module):
    """GraphActorCritic: A class representing the Actor-Critic architecture for graph-based reinforcement learning.

    This class implements a graph-based Actor-Critic architecture for reinforcement learning. It takes as input a
    graph neural network (gnn) and applies a set of fully-connected layers to compute the action logits and the
    value function.

    Args:
        gnn (nn.Module): A graph neural network to extract features from the input graph.
        action_refers_to (str): A string indicating whether the actions should be taken on nodes ('node') or edges ('edge').
        pool_type (str): A string indicating the type of graph pooling to be applied.
        action_distr (str): A string indicating the type of action distribution to use.
        act_fn (str): A string indicating the activation function to use in the network.
        init_fn (callable, optional): A function to initialize the network weights. Defaults to None.
    """

    def __init__(
        self,
        policy: GraphActorCritic,
        eps_clip: float = 0.2,
        gamma: float = 0.99,
        coeff_mse: float = 0.0,
        coeff_entropy: float = 0.0,
    ):
        self.eps_clip = eps_clip

        self.gamma = gamma

        self.coeff_mse = coeff_mse
        self.coeff_entropy = coeff_entropy

        super(GraphPPO, self).__init__()

        self.buffer = RolloutBuffer()
        self.policy = policy

        self.policy_old = copy.deepcopy(policy)

        self.mse_loss = nn.MSELoss(reduction="none")

    def get_optimization_config(self, lr_actor: float, lr_critic: float):
        return [
            {"params": self.policy.actor_params(), "lr": lr_actor},
            {"params": self.policy.critic_params(), "lr": lr_critic},
        ]

    @torch.no_grad()
    def act(
        self,
        state: pygd.Batch,
        return_logprobs: bool = False,
        sample: bool = True,
        values: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return_list = []

        if sample:
            return_list.append("action_sample")
        else:
            return_list.append("action_mode")

        if return_logprobs:
            return_list.append("action_logprobs")

        if values:
            return_list.append("state_values")

        output_dict = self.policy_old.compute(state, return_list=return_list, detach=True)

        # Rename key action_sample to action
        if "action_sample" in output_dict:
            output_dict["action"] = output_dict.pop("action_sample")
        elif "action_mode" in output_dict:
            output_dict["action"] = output_dict.pop("action_mode")

        return output_dict

    def forward(self, shuffle: bool = False):
        assert not shuffle
        assert self.policy.training
        assert self.policy.gnn_actor.training
        assert self.policy.gnn_critic.training
        assert self.policy.actor.training
        assert self.policy.critic.training

        (
            old_states,
            old_actions,
            old_logprobs,
            rewards_norm,
            rewards,
            advantages,
        ) = self.buffer.experiences

        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

        old_logprobs = old_logprobs.to(logprobs.device)
        rewards_norm = rewards_norm.to(logprobs.device)
        rewards = rewards.to(logprobs.device)
        advantages = advantages.to(logprobs.device)

        action_mean = self.policy.action_distr.mean()
        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values)

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - old_logprobs.detach())

        # Finding Surrogate Loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

        diff_surr = surr1 - surr2
        # final loss of clipped objective PPO
        my_surr = torch.min(surr1, surr2)

        my_mse = self.mse_loss(state_values, rewards_norm)

        my_mse_coeff = self.coeff_mse * my_mse

        objective = my_surr - my_mse_coeff + self.coeff_entropy * dist_entropy

        loss = -objective

        loss_dict = {
            "loss": loss,
            "surrogate": my_surr.detach(),
            "diff_surr": diff_surr.detach(),
            "ratios": ratios.detach(),
            "actions_mu": action_mean.detach(),
            "mse": my_mse.detach(),
            "mse_coeff": my_mse_coeff.detach(),
            "entropy": dist_entropy.detach(),
            "rewards": rewards.detach(),
        }

        return loss_dict

    @torch.no_grad()
    def prepare_forward(self, env: GraphEnvBase, n_steps: int = 10) -> None:
        done = True
        state = None

        idx_list = list(range(len(env.loader)))

        random.shuffle(idx_list)

        if len(idx_list) < n_steps:
            logging.warning(
                f"Number of graphs {len(idx_list)} in the dataset is less than the number of steps requested. "
            )
        for idx in idx_list[:n_steps]:
            if done:
                state = env.reset(idx=idx)
            act_dict = self.act(state, return_logprobs=True, sample=True, values=True)
            action = act_dict["action"]
            logprobs = act_dict["action_logprobs"]
            state_values = act_dict["state_values"]

            self.buffer.append(state=state, state_value=state_values)
            state, reward, done, info = env.step(action)

            self.buffer.append(reward=reward, done=done, action=action, action_logprob=logprobs)

        self.buffer.prepare(gamma=self.gamma, device="cpu")

    def forward_end(self) -> None:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    @torch.no_grad()
    def run_episode(
        self, batch: pygd.Batch, env: GraphEnvBase, sample: bool = True, num_samples: int = 1
    ) -> pygd.Batch:
        data_list = batch.to_data_list()
        batch_out = []

        assert not self.policy.training
        assert not self.policy_old.training

        for graph_i in data_list:
            for _ in range(num_samples):
                state = env.reset(graph=graph_i)
                done = False
                while not done:
                    act_dict = self.act(state=state, sample=sample)
                    action = act_dict["action"]
                    state, _, done, info = env.step(action=action)

                graph_out = env.get_final_graph()
                delattr(graph_out, "batch")

                graph_out.action = action
                batch_out.append(graph_out)

        batch = pygd.Batch.from_data_list(data_list=batch_out)
        batch.batch = batch.batch

        env.on_episode_end()

        return batch
