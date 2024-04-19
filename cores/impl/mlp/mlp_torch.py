from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from cores.core.values.constants import ActivationTypes
from cores.impl.activation.activation_torch import get_activation as act_fn
from cores.utils.exceptions import WrongDimensions


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers_num: int,
        activation: ActivationTypes,
        device: str,
        hidden_dim: Optional[int] = None,
        has_bn: bool = False,
        has_ln: bool = False,
        use_act_out: bool = False,
        dropout: float = 0.0,
        drop_last: bool = True,
    ):
        super(MLP, self).__init__()
        assert layers_num >= 0
        self.device = device
        self.output_dim = output_dim

        layers = []
        for n in range(layers_num):
            input_dim_i = None
            output_dim_i = None
            blocks = []
            if n == 0:
                if layers_num == 1:
                    input_dim_i = input_dim
                    output_dim_i = output_dim
                    act = act_fn(activation) if use_act_out else nn.Identity().to(self.device)
                else:
                    input_dim_i = input_dim
                    output_dim_i = hidden_dim
                    act = act_fn(activation)
            elif n == (layers_num - 1):
                input_dim_i = hidden_dim
                output_dim_i = output_dim
                act = act_fn(activation) if use_act_out else nn.Identity().to(self.device)
            else:
                input_dim_i = hidden_dim
                output_dim_i = hidden_dim
                act = act_fn(activation)

            blocks.append(nn.Linear(input_dim_i, output_dim_i, device=self.device))
            if has_bn:
                blocks.append(nn.BatchNorm1d(output_dim_i, device=self.device))
            elif has_ln:
                blocks.append(nn.LayerNorm(output_dim_i, device=self.device))
            blocks.append(act)

            if dropout > 0.0 and (n < (layers_num - 1) or drop_last):
                drop = nn.Dropout(dropout).to(device)
                blocks.append(drop)
            layers.append(nn.Sequential(*blocks).to(self.device))

        if layers_num == 0:
            layers = [nn.Identity().to(device)]
            if input_dim != output_dim:
                raise WrongDimensions(input_dim, output_dim)

        self.fc_layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor):

        for fc in self.fc_layers:
            x = fc(x)

        return x
