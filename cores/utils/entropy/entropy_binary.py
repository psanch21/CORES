from __future__ import annotations

import torch

import cores.core.values.typing as gtp
from cores.utils.entropy.entropy_base import Entropy


class BinaryEntropy(Entropy):
    def __init__(self):
        return

    def _compute(self, logits: torch.FloatTensor) -> gtp.Tensor1D:
        assert logits.ndim <= 2
        if logits.ndim == 2:
            assert logits.shape[-1] == 1
            logits = logits.flatten()

        pred = torch.sigmoid(logits).clamp(min=1e-5, max=1.0 - 1e-5)
        pred_1 = 1.0 - pred

        entropy = -pred * torch.log(pred) - pred_1 * torch.log(pred_1)
        return entropy

    def entropy_max(self, logits: torch.FloatTensor) -> gtp.Tensor0D:
        return -torch.log(torch.tensor(0.5))
