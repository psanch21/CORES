from __future__ import annotations


from typing import Tuple

import numpy as np
import numpy.typing as ntp
import torch
from scipy import stats

import cores.core.values.typing as gtp
from cores.utils.conformal.conformal_base import ConformalPrediction


class ConformalTorch(ConformalPrediction):
    def coverage_distribution(
        self,
        x_cal: gtp.Tensor2D,
        y_cal: gtp.Tensor2D,
        x_val: gtp.Tensor2D,
        y_val: gtp.Tensor2D,
        iterations: int = 10,
    ) -> gtp.Tensor1D:
        x = torch.cat([x_cal, x_val])
        y = torch.cat([y_cal, y_val])

        scores = self.score_fn(x, y)

        n = self.calibration_num

        coverages = torch.zeros(iterations)

        for r in range(iterations):
            idx_rnd = torch.randperm(scores.shape[0])

            calib_scores = scores[idx_rnd[:n]]
            val_scores = scores[idx_rnd[n:]]

            qhat = self.compute_quantile(calib_scores)

            coverage = (val_scores <= qhat).float().mean()
            coverages[r] = coverage

        return coverages

    def sample_coverage_distribution(self, samples_num: int) -> gtp.Tensor1D:
        l = np.floor((self.calibration_num + 1) * self.alpha)
        alpha = self.calibration_num + 1 - l
        beta = l

        known_dist = stats.beta(alpha, beta)

        samples = known_dist.rvs(samples_num)
        return torch.tensor(samples)

    def coverage_ks_test(
        self, coverages: ntp.NDArray, significance_level: float = 0.05
    ) -> Tuple[bool, float]:
        l = np.floor((self.calibration_num + 1) * self.alpha)
        alpha = self.calibration_num + 1 - l
        beta = l

        known_dist = stats.beta(alpha, beta)

        coverages = coverages.cpu().numpy()

        ks_statistic, ks_p_value = stats.kstest(
            coverages, known_dist.cdf, N=len(coverages), method="exact"
        )
        success = ks_p_value >= significance_level
        return success, ks_p_value

    def fit(self, u_values: gtp.Tensor2D, y: gtp.Tensor2D) -> None:
        self.calibration_num = u_values.shape[0]
        scores = self.score_fn(u_values, y)
        self.q_hat = self.compute_quantile(scores)

    def compute_quantile(self, scores: gtp.Tensor1D) -> gtp.Tensor0D:
        assert scores.ndim == 1
        q = np.ceil((self.calibration_num + 1) * (1 - self.alpha)) / self.calibration_num
        q = min(q, 1.0)
        quantile = torch.quantile(scores, q, interpolation="higher")
        return quantile
