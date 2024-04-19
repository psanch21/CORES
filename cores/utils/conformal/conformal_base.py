from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from cores.core.values.typing import Tensor1D, Tensor2D


class ConformalPrediction(ABC):
    """
    Conformal Prediction
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.calibration_num = None
        self.q_hat = None

    @abstractmethod
    def score_fn(self, x: Tensor2D, y: Optional[Tensor1D] = None) -> Tensor1D | Tensor2D:
        pass

    @abstractmethod
    def coverage_distribution(
        self,
        x_cal: Tensor2D,
        y_cal: Tensor1D,
        x_val: Tensor2D,
        y_val: Tensor1D,
        iterations: int = 10,
    ):
        pass

    @abstractmethod
    def coverage_ks_test(
        self, coverages: Tensor1D, significance_level: float = 0.05
    ) -> Tuple[bool, float]:
        pass

    @abstractmethod
    def fit(self, x: Tensor2D, y: Tensor1D) -> None:
        pass

    @abstractmethod
    def compute_quantile(self, scores: Tensor1D):
        pass

    @abstractmethod
    def compute_conformal_set(self, u_values: Tensor2D) -> None:
        pass

    def __call__(self, u_values: Tensor2D) -> Tensor2D:
        conformal_set = self.compute_conformal_set(u_values)

        return conformal_set
