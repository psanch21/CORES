from __future__ import annotations

import torch

import cores.core.values.typing as gtp
from cores.utils.conformal.conformal_torch import ConformalTorch
from cores.utils.torch import TorchUtils


class ConformalAPSTorch(ConformalTorch):
    def __init__(self, *args, classes_num: int, **kwargs):
        self.classes_num = classes_num
        super().__init__(*args, **kwargs)

    def score_fn(self, u_values: gtp.Tensor2D, y: gtp.Tensor2D) -> gtp.Tensor2D:
        assert u_values.shape[1] == self.classes_num, f"u_values.shape: {u_values.shape}"
        assert y.ndim == 1
        assert y.shape[0] == u_values.shape[0]

        assert torch.allclose(u_values.sum(1), torch.tensor(1.0, dtype=torch.float32))

        idx_ordered, u_sorted, u_sorted_cumsum = TorchUtils.sort_sum(u_values)

        idx_y = torch.nonzero(idx_ordered == y.view(-1, 1))[:, 1]

        batch_size = u_values.shape[0]
        scores = u_sorted_cumsum[torch.arange(batch_size), idx_y]

        return scores

    def compute_conformal_set(self, u_values: gtp.Tensor2D) -> gtp.Tensor2D:
        batch_size = u_values.shape[0]

        device = u_values.device
        assert u_values.shape[1] == self.classes_num
        assert torch.allclose(u_values.sum(1), torch.tensor(1.0, dtype=u_values.dtype))
        y_set = torch.zeros(batch_size, self.classes_num).to(device)
        y_argmax = u_values.argmax(dim=1)
        y_set[torch.arange(batch_size), y_argmax] = 1
        for i in range(self.classes_num):
            y_i = torch.full((batch_size,), i).to(device)

            scores_i = self.score_fn(u_values, y_i)

            y_set[scores_i < self.q_hat, i] = 1

        return y_set
