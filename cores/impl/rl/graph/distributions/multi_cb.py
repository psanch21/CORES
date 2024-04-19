from __future__ import annotations

import torch
import torch as th
from torch import nn
from torch.distributions import ContinuousBernoulli

from cores.utils import TorchUtils

from .multi import MultiDistribution


class MultiGraphContBernoulliDistribution(MultiDistribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self):
        super(MultiGraphContBernoulliDistribution, self).__init__()

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
    ) -> MultiGraphContBernoulliDistribution:
        assert action_logits.shape[1] == 1, "The dimension of each node should be 1"
        self.num_nodes_per_graph = tuple(
            TorchUtils.scatter_sum(src=torch.ones_like(batch_idx), index=batch_idx)
        )
        action_logits_splits = th.split(
            action_logits.flatten(), split_size_or_sections=self.num_nodes_per_graph
        )
        self.distribution = [ContinuousBernoulli(logits=split) for split in action_logits_splits]
        return self

    def mode(self) -> th.Tensor:
        return th.cat([dist.probs.round() for dist in self.distribution])
