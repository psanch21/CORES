from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import stats

from cores.utils.conformal.conformal_base import ConformalPrediction


class ConformalNP(ConformalPrediction):
    def coverage_distribution(
        self,
        x_cal: npt.NDArray,
        y_cal: npt.NDArray,
        x_val: npt.NDArray,
        y_val: npt.NDArray,
        iterations: int = 10,
    ) -> npt.NDArray:
        x = np.concatenate([x_cal, x_val])
        y = np.concatenate([y_cal, y_val])

        scores = self.score_fn(x, y)

        n = self.calibration_num

        coverages = np.zeros(iterations)

        for r in range(iterations):
            random.shuffle(scores)

            calib_scores, val_scores = (scores[:n], scores[n:])

            qhat = self.compute_quantile(calib_scores)

            coverage = (val_scores <= qhat).mean()
            coverages[r] = coverage

        return coverages

    def sample_coverage_distribution(self, samples_num: int) -> npt.NDArray:
        l = np.floor((self.calibration_num + 1) * self.alpha)
        alpha = self.calibration_num + 1 - l
        beta = l

        known_dist = stats.beta(alpha, beta)

        samples = known_dist.rvs(samples_num)
        return samples

    def coverage_ks_test(
        self, coverages: npt.NDArray, significance_level: float = 0.05
    ) -> Tuple[bool, float]:
        l = np.floor((self.calibration_num + 1) * self.alpha)
        alpha = self.calibration_num + 1 - l
        beta = l

        known_dist = stats.beta(alpha, beta)

        ks_statistic, ks_p_value = stats.kstest(
            coverages, known_dist.cdf, N=len(coverages), method="exact"
        )
        success = ks_p_value >= significance_level
        return success, ks_p_value

    def fit(self, u_values: npt.NDArray, y: npt.NDArray) -> None:
        self.calibration_num = u_values.shape[0]
        scores = self.score_fn(u_values, y)
        self.q_hat = self.compute_quantile(scores)

    def compute_quantile(self, scores: npt.NDArray) -> float:
        assert scores.ndim == 1
        q = np.ceil((self.calibration_num + 1) * (1 - self.alpha)) / self.calibration_num
        q = min(q, 1.0)
        return np.quantile(scores, q, method="higher")
