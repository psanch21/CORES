from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.nn import TopKPooling
from torch_geometric.typing import OptTensor


class MyTopKPooling(TopKPooling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, OptTensor, OptTensor, Tensor, Tensor]:
        r"""
        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
            attn (torch.Tensor, optional): Optional node-level matrix to use
                for computing attention scores instead of using the node
                feature matrix :obj:`x`. (default: :obj:`None`)
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        select_out = self.select(attn, batch)

        perm = select_out.node_index
        score = select_out.weight
        assert score is not None

        # Sort perm and return new indices
        perm_idx = torch.argsort(perm)
        perm = perm[perm_idx]
        score = score[perm_idx]

        return (perm, score)

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f"ratio={self.ratio}"
        else:
            ratio = f"min_score={self.min_score}"

        return (
            f"{self.__class__.__name__}({self.in_channels}, {ratio}, "
            f"multiplier={self.multiplier})"
        )
