from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

import cores.core.values.constants as cte


class TorchUtils:
    @staticmethod
    def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand(other.size())
        return src

    @staticmethod
    def scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)

    @staticmethod
    def scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        index = TorchUtils.broadcast(index, src, dim)
        if out is None:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() == 0:
                size[dim] = 0
            else:
                size[dim] = int(index.max()) + 1
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
            return out.scatter_add_(dim, index, src)
        else:
            return out.scatter_add_(dim, index, src)

    @staticmethod
    def split_into_segments(original_len: int, segment_sizes: List[float], k_fold: int = -1):
        all_idx = torch.arange(original_len)
        if len(segment_sizes) == 1:
            return [all_idx]
        if k_fold >= 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(k_fold)
            perm = torch.randperm(original_len, generator=generator)
            all_idx = all_idx[perm]
            n = len(perm) * (1 - segment_sizes[0])
            all_idx = torch.roll(all_idx, shifts=int(n * k_fold))
        start_idx, end_idx = 0, None
        all_idx_splits = []

        num_splits = len(segment_sizes)
        for i, size in enumerate(segment_sizes):
            assert isinstance(size, float)
            assert 0 < size
            assert 1 > size
            new_len = int(size * original_len)
            end_idx = new_len + start_idx
            if i == (num_splits - 1):
                all_idx_splits.append(all_idx[start_idx:])
            else:
                all_idx_splits.append(all_idx[start_idx:end_idx])
            start_idx = end_idx

        return all_idx_splits

    @staticmethod
    def stratify(
        y: torch.LongTensor, original_len: int, segment_sizes: List[float], k_fold: int = -1
    ) -> List[List[int]]:
        all_idx = list(range(original_len))

        if k_fold == -1:
            random_state = None
        else:
            random_state = k_fold

        all_idx_splits = []

        test_size = 1.0

        for i, segment_size in enumerate(segment_sizes):
            test_size = sum(segment_sizes[(i + 1) :]) / sum(segment_sizes[i:])
            test_size = max(test_size, 0.0)
            if test_size > 0.0:
                idx_i, all_idx, _, y = train_test_split(
                    all_idx, y, test_size=test_size, random_state=random_state, stratify=y
                )
                all_idx_splits.append(idx_i)
            else:
                all_idx_splits.append(all_idx)

        return all_idx_splits

    @staticmethod
    def build_optimizer(
        name: cte.OptimizerType, optim_kwargs: Dict[str, Any], params: nn.ParameterList
    ) -> optim.Optimizer:
        params = filter(lambda p: p.requires_grad, params)
        # Try to load customized optimizer

        if name == cte.OptimizerType.ADAM:
            optimizer = optim.Adam(params, **optim_kwargs)
        elif name == cte.OptimizerType.RADAM:
            optimizer = optim.RAdam(params, **optim_kwargs)
        elif name == cte.OptimizerType.SGD:
            optimizer = optim.SGD(params, **optim_kwargs)
        else:
            raise ValueError(f"Optimizer {name} not supported")

        return optimizer

    @staticmethod
    def build_scheduler(
        name: cte.LRSchedulerType, scheduler_kwargs: Dict[str, Any], optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LRScheduler:
        # Try to load customized scheduler

        if name == cte.LRSchedulerType.STEP:
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
        elif name == cte.LRSchedulerType.EXP:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_kwargs)
        elif name == cte.LRSchedulerType.COS:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
        elif name == cte.LRSchedulerType.PLATEAU:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_kwargs,
            )
        else:
            raise ValueError(f"Scheduler {name} not supported")
        return scheduler

    @staticmethod
    def sort_sum(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        sorted_scores, indexes = torch.sort(scores, dim=1, descending=True)
        scores_ordered_cumsum = torch.cumsum(sorted_scores, dim=1)

        return indexes, sorted_scores, scores_ordered_cumsum

    @staticmethod
    def expand_binary_probs(probs: torch.Tensor) -> torch.Tensor:
        assert probs.ndim == 2

        if probs.shape[1] == 1:
            assert probs.min() >= 0
            assert probs.max() <= 1
            probs_0 = 1.0 - probs

            probs = torch.cat([probs_0, probs], dim=1)
            return probs
        else:
            assert torch.allclose(probs.sum(1), torch.tensor(1.0, dtype=probs.dtype))
            return probs
