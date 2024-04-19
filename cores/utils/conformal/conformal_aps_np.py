from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cores.utils.conformal.conformal_np import ConformalNP
from cores.utils.numpy import NPUtils


class ConformalAPSNP(ConformalNP):
    def __init__(self, *args, classes_num: int, **kwargs):
        self.classes_num = classes_num
        super().__init__(*args, **kwargs)

    def score_fn(self, u_values: npt.NDArray, y: npt.NDArray) -> npt.NDArray | npt.NDArray:
        assert u_values.shape[1] == self.classes_num
        assert y.ndim == 1
        assert y.shape[0] == u_values.shape[0]

        idx_ordered, u_sorted, u_sorted_cumsum = NPUtils.sort_sum(u_values)

        idx_y = np.where(idx_ordered == y.reshape(-1, 1))[1]

        batch_size = u_values.shape[0]
        scores = u_sorted_cumsum[np.arange(batch_size), idx_y]

        return scores

    def compute_conformal_set(self, u_values: npt.NDArray) -> npt.NDArray:
        batch_size = u_values.shape[0]
        y_set = np.zeros((batch_size, self.classes_num))
        y_argmax = u_values.argmax(axis=1)
        y_set[np.arange(batch_size), y_argmax] = 1
        for i in range(self.classes_num):
            y_i = np.ones(batch_size) * i
            scores_i = self.score_fn(u_values, y_i)

            y_set[scores_i <= self.q_hat, i] = 1

        return y_set
