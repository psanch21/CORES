from __future__ import annotations
from typing import Tuple

import torch as th
from torch import nn

from .base import Distribution


class MultiDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self):
        super(MultiDistribution, self).__init__()
        self.num_nodes_per_graph = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        raise NotImplementedError

    def proba_distribution(
        self, action_logits: th.Tensor, batch_idx: th.Tensor
    ) -> MultiDistribution:
        raise NotImplementedError

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        action_splits = th.split(actions.flatten(), split_size_or_sections=self.num_nodes_per_graph)

        return th.stack(
            [dist.log_prob(action).sum() for dist, action in zip(self.distribution, action_splits)]
        )

    def entropy(self) -> th.Tensor:
        return th.stack([dist.entropy().sum() for dist in self.distribution], dim=0)

    def sample(self) -> th.Tensor:
        return th.cat([dist.sample() for dist in self.distribution])

    def mode(self) -> th.Tensor:
        raise NotImplementedError

    def mean(self) -> th.Tensor:
        return th.cat([dist.mean for dist in self.distribution])

    def actions_from_params(
        self, action_logits: th.Tensor, batch_idx: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits, batch_idx=batch_idx)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, action_logits: th.Tensor, batch_idx: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits, batch_idx=batch_idx)
        log_prob = self.log_prob(actions)
        return actions, log_prob
