from __future__ import annotations

import torch

import cores.core.values.typing as gtp
from cores.utils.entropy.entropy_base import Entropy


class CategoricalEntropy(Entropy):
    def __init__(self):
        return

    def _compute(self, logits: torch.FloatTensor) -> gtp.Tensor1D:
        is_flat = logits.ndim == 1
        if is_flat:
            logits = logits.unsqueeze(0)
        pred = torch.softmax(logits, dim=-1).clamp(min=1e-5, max=1.0 - 1e-5)
        log_pred = torch.log(pred)
        entropy = -torch.sum(pred * log_pred, dim=-1)
        if is_flat:
            entropy = entropy.flatten()
        return entropy

    def entropy_max(self, logits: torch.FloatTensor) -> gtp.Tensor0D:
        if logits.ndim == 1:
            K = logits.shape[0]
        else:
            K = logits.shape[1]
        entropy_max = -torch.log(1.0 / torch.tensor(K))

        return entropy_max
