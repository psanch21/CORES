from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Entropy(ABC):
    def __init__(self):
        pass

    def __call__(self, logits: torch.FloatTensor, normalize: bool = False):
        entropy = self._compute(logits)

        if normalize:
            entropy_max = self.entropy_max(logits)
            return entropy / entropy_max
        else:
            return entropy

    @abstractmethod
    def _compute(self, logits: torch.FloatTensor):
        pass

    @abstractmethod
    def entropy_max(self, logits: torch.FloatTensor):
        pass
