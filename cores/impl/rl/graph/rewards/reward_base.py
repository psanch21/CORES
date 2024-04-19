from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch_geometric.data as pygd

from cores.impl.gnn.pyg.gnn_base import BaseGNN


class Reward(ABC):
    def __init__(
        self,
        graph_clf: BaseGNN,
        classes_num: int,
        loss_fn: Callable,
        device: str = "cpu",
    ):
        self.graph_clf = graph_clf

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.classes_num = classes_num

        self.loss_fn = loss_fn

        self.state_0 = None
        self.target = None
        self.logits_0 = None

    def batch_to_target(self, batch: pygd.Data):
        return batch.y

    def logits_to_soft_pred(self, logits: torch.FloatTensor):
        if self.classes_num > 2:
            return torch.softmax(logits, dim=-1)
        else:
            return torch.sigmoid(logits)

    def logits_to_hard_pred(self, logits: torch.FloatTensor):
        if self.classes_num > 2:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            return (logits > 0).long()

    def full_graph_is_correct(self):
        target_clf = self.logits_to_hard_pred(self.logits_0)
        return (self.target == target_clf).float().mean() > 0.5

    @abstractmethod
    def _compute(self, logits: torch.FloatTensor, state: pygd.Batch, **kwargs):
        pass

    def compute(self, state: pygd.Batch, **kwargs):
        if self.state_0 is None:
            raise ValueError("state_0 is None. Call set_state_0() first.")
        logits = self.graph_clf(state.clone().to(self.device))
        assert logits.ndim == 2
        if logits.shape[0] > 1:
            raise NotImplementedError("num_graphs > 1 not implemented yet")
        reward = self._compute(logits, state=state.clone(), **kwargs)
        assert reward.ndim == 2, f"logits: {logits.shape}, reward: {reward.shape}"
        assert reward.shape[-1] == 1
        assert reward.shape[0] == 1
        reward = reward.flatten()

        return reward

    def _num_samples(self, logits: torch.FloatTensor):
        return logits.shape[0]

    def _subgraph_is_correct(self, pred_hard: torch.LongTensor):
        assert pred_hard.numel() == 1
        assert self.target.numel() == 1
        is_correct = (pred_hard == self.target).float()
        return is_correct.mean() > 0.5

    def fit_conformal_from_loader(
        self, loader: Optional[pygd.DataLoader] = None, batch_norm_fn: Optional[Callable] = None
    ):
        return

    def set_state_0(self, state_0: pygd.Batch):
        if state_0.num_edges == 0:
            raise ValueError("state_0 has no edges")
        self.state_0 = state_0
        self.target = self.batch_to_target(state_0).flatten()
        logits = self.graph_clf(state_0.clone().to(self.device))
        assert logits.ndim == 2

        if logits.shape[0] > 1:
            raise NotImplementedError("num_graphs > 1 not implemented yet")
        self.logits_0 = logits

    def __str__(self):
        my_str = f"{self.__class__}\n"
        return my_str
