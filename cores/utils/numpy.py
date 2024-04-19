from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
from sklearn.utils import resample


class NPUtils:
    @staticmethod
    def resample(
        target: npt.NDArray, upsample: bool = True, replace: bool = True, random_state: int = 42
    ) -> npt.NDArray:
        # Determine the number of samples in the majority class
        class_counts = np.bincount(target)
        if upsample:
            class_count = np.max(class_counts)
        else:
            class_count = np.min(class_counts)

        # Create empty arrays to store the upsampled data
        idx_list = []

        # Iterate over each class
        for class_label in np.unique(target):
            # Get the indices of samples belonging to the current class
            indices = np.where(target == class_label)[0]

            # Check if the class is a minority class
            if len(indices) != class_count:
                # Upsample the minority class to match the majority class size
                indices_resampled = resample(
                    indices,
                    n_samples=class_count,
                    replace=replace,
                    random_state=random_state,
                )

                # Append the upsampled data to the arrays
                idx_list.extend(indices_resampled)
            else:
                # If the class is not a minority class, just append the data as is
                idx_list.extend(indices)
        # Convert the upsampled data back to NumPy arrays
        indices_new = np.array(idx_list)

        return indices_new

    @staticmethod
    def sort_sum(scores: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Sorts the scores and computes the cumulative sum of the sorted scores.
        Args:
            scores: The scores to be sorted.

        Returns:
            The indexes of the sorted scores, the sorted scores
            and the cumulative sum of the sorted scores.
        """
        assert scores.ndim == 2
        assert scores.shape[1] > 1

        indexes = scores.argsort(axis=1)[:, ::-1]
        scores_ordered = np.sort(scores, axis=1)[:, ::-1]
        scores_ordered_cumsum = np.cumsum(scores_ordered, axis=1)
        return indexes, scores_ordered, scores_ordered_cumsum

    @staticmethod
    def expand_binary_probs(probs: npt.NDArray) -> npt.NDArray:
        assert probs.ndim == 2

        if probs.shape[1] == 1:
            probs_0 = 1.0 - probs

            probs = np.concatenate([probs_0, probs], axis=1)
            return probs
        else:
            return probs
