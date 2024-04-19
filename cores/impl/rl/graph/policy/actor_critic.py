from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch_geometric.data as pygd

import cores.core.values.constants as cte
import cores.core.values.typing as gtp
from cores.impl.activation.activation_torch import get_activation
from cores.impl.gnn.pyg.gnn_base import BaseGNN
from cores.impl.gnn.pyg.pooling import GraphPooling
from cores.impl.rl.graph.distributions import (
    MultiGraphBernoulliDistribution,
    MultiGraphContBernoulliDistribution,
)


class GraphActorCritic(nn.Module):
    """GraphActorCritic: A class representing the Actor-Critic architecture for graph-based reinforcement learning."""

    def __init__(
        self,
        gnn: BaseGNN,
        action_refers_to: cte.ActionTypes,
        pool_type: List[cte.PoolingTypes] | cte.PoolingTypes,
        action_distr: cte.Distributions,
        activation: cte.ActionTypes,
    ):
        super(GraphActorCritic, self).__init__()

        if isinstance(activation, str):
            activation = cte.ActivationTypes(activation)

        hidden_dim = gnn.output_dim

        self.action_refers_to = action_refers_to

        self.gnn_actor = gnn
        self.device = gnn.device

        self.gnn_critic = copy.deepcopy(gnn)
        dim = hidden_dim if self.action_refers_to == cte.ActionTypes.NODE else 2 * hidden_dim

        self.actor = nn.Sequential(
            nn.Linear(dim, 1, device=self.device),
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1, device=self.device),
        ).to(self.device)

        self.act_fn = get_activation(activation)

        self.graph_pooling = GraphPooling(
            pool_type=pool_type,
            in_channels=hidden_dim,
            activation=activation,
            bn=False,
        )

        if action_distr == cte.Distributions.BERNOULLI:
            self.action_distr = MultiGraphBernoulliDistribution()
        elif action_distr == cte.Distributions.CONT_BERNOULLI:
            self.action_distr = MultiGraphContBernoulliDistribution()
        else:
            raise NotImplementedError(f"Unknown distribution {action_distr}.")

    def _get_attr(
        self, batch: pygd.Batch, attr: str, refers_to: cte.ActionTypes = cte.ActionTypes.NODE
    ) -> torch.Tensor:
        my_attr = getattr(batch, attr)

        if attr == "batch" and refers_to == cte.ActionTypes.EDGE:
            edge_index = getattr(batch, "edge_index")
            my_attr = my_attr[edge_index[0]]

        return my_attr

    def _get_edge_attr(self, state: pygd.Batch) -> torch.Tensor:
        if hasattr(state, "edge_attr"):
            edge_attr = self._get_attr(state, "edge_attr")
        elif hasattr(state, "edge_feature"):
            edge_attr = self._get_attr(state, "edge_feature")
        else:
            edge_attr = None

        return edge_attr

    def actor_params(self) -> nn.ParameterList:
        params = list(self.actor.parameters())
        params.extend(list(self.gnn_actor.parameters()))
        return nn.ParameterList(params)

    def critic_params(self) -> nn.ParameterList:
        params = list(self.critic.parameters())
        params.extend(list(self.gnn_critic.parameters()))
        return nn.ParameterList(params)

    def feature_extractor_actor(self, batch: pygd.Batch, **kwargs):
        logits_actor = self.gnn_actor(batch.clone(), **kwargs)

        logits_actor = self.act_fn(logits_actor)
        return logits_actor

    def feature_extractor_critic(self, batch: pygd.Batch, **kwargs) -> torch.Tensor:
        logits_critic = self.gnn_critic(batch.clone(), **kwargs)

        logits_critic = self.act_fn(logits_critic)

        return logits_critic

    def forward(self):
        raise NotImplementedError

    def get_action_logits_from_features(
        self, z: torch.Tensor, edge_index: torch.LongTensor
    ) -> torch.FloatTensor:
        if self.action_refers_to == cte.ActionTypes.EDGE:
            zl = z[edge_index[0]]
            zr = z[edge_index[1]]
            z2 = torch.cat([zl, zr], dim=1)
            action_logits = self.actor(z2)
        else:
            action_logits = self.actor(z)

        return action_logits - 1.0

    def compute_action_logits(self, state: pygd.Batch) -> torch.FloatTensor:
        edge_index = self._get_attr(state, "edge_index")

        z_actor = self.feature_extractor_actor(batch=state)

        action_logits = self.get_action_logits_from_features(z=z_actor, edge_index=edge_index)
        return action_logits

    def act(
        self, state: pygd.Batch, sample: bool = True
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        action_logits = self.compute_action_logits(state=state)
        batch = self._get_attr(state, "batch", refers_to=self.action_refers_to)

        self.action_distr.proba_distribution(action_logits=action_logits, batch_idx=batch)
        if sample:
            actions = self.action_distr.sample()
        else:
            actions = self.action_distr.mode()

        action_logprobs = self.action_distr.log_prob(actions)
        return actions.detach(), action_logprobs.detach()

    def _graph_pooling(self, z: torch.FloatTensor, batch: torch.LongTensor):
        z = self.graph_pooling.forward2(x=z, batch=batch)
        return z

    def compute(
        self, state: pygd.Batch, return_list: List[str] = None, detach: bool = True
    ) -> Dict[str, torch.Tensor]:
        if return_list is None:
            # action_sample, action_mode, action_logprobs, values
            return_list = ["action_sample", "action_logprobs", "state_values"]
        state = state.to(self.device)
        edge_index = self._get_attr(state, "edge_index")

        batch = self._get_attr(state, "batch", refers_to=self.action_refers_to)

        z_actor = self.feature_extractor_actor(batch=state)

        output = {}

        compute_action_logits = any(
            [a in return_list for a in ["action_sample", "action_mode", "action_logprobs"]]
        )

        if compute_action_logits:
            action_logits = self.get_action_logits_from_features(z=z_actor, edge_index=edge_index)

            self.action_distr.proba_distribution(action_logits=action_logits, batch_idx=batch)

            if "action_sample" in return_list:
                actions = self.action_distr.sample()
                output["action_sample"] = actions
            elif "action_mode" in return_list:
                actions = self.action_distr.mode()
                output["action_mode"] = actions

            if "action_logprobs" in return_list:
                action_logprobs = self.action_distr.log_prob(actions)

                output["action_logprobs"] = action_logprobs

        if "state_values" in return_list:
            z_critic = self.feature_extractor_critic(batch=state)
            z_pool = self._graph_pooling(z=z_critic, batch=state.batch)

            state_values = self.critic(z_pool)
            # pb_io.print_debug_tensor(state_values, 'state_values')
            output["state_values"] = state_values

        if detach:
            for key, value in output.items():
                output[key] = value.detach()
        return output

    def get_state_values(self, state: pygd.Batch) -> torch.FloatTensor:
        z = self.feature_extractor_critic(batch=state)
        batch = self._get_attr(state, "batch")

        z_pool = self._graph_pooling(z=z, batch=batch)

        state_values = self.critic(z_pool)

        return state_values

    def evaluate(
        self, state: pygd.Batch, action: torch.FloatTensor
    ) -> Tuple[gtp.Tensor1D, gtp.Tensor2D, gtp.Tensor1D]:
        state = state.to(self.device)
        action = action.to(self.device)
        edge_index = self._get_attr(state, "edge_index")

        z_actor = self.feature_extractor_actor(batch=state)

        action_logits = self.get_action_logits_from_features(z=z_actor, edge_index=edge_index)

        batch = self._get_attr(state, "batch", refers_to=self.action_refers_to)

        self.action_distr.proba_distribution(action_logits=action_logits, batch_idx=batch)
        action_logprobs = self.action_distr.log_prob(action)
        dist_entropy = self.action_distr.entropy()

        batch = self._get_attr(state, "batch")
        z_critic = self.feature_extractor_critic(batch=state)
        z_pool = self._graph_pooling(z=z_critic, batch=batch)
        state_values = self.critic(z_pool)
        if "cuda" in action_logprobs.device.type:
            action_logprobs = action_logprobs.to("cpu")
        if "cuda" in state_values.device.type:
            state_values = state_values.to("cpu")
        if "cuda" in dist_entropy.device.type:
            dist_entropy = dist_entropy.to("cpu")
        return action_logprobs, state_values, dist_entropy
