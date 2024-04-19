from __future__ import annotations

import torch
import torch.nn as nn

from cores.core.values.constants import ActivationTypes


class SinActivation(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()  # init the base class

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the function.
        """
        return torch.sin(input)  # simply apply already implemented SiLU


def get_activation(name: ActivationTypes, **kwargs):
    if name == ActivationTypes.RELU:
        return nn.ReLU(inplace=False)
    elif name == ActivationTypes.TANH:
        return nn.Tanh()
    elif name == ActivationTypes.SELU:
        return nn.SELU(inplace=False)
    elif name == ActivationTypes.SIGMOID:
        return nn.Sigmoid()
    elif name == ActivationTypes.PRELU:
        return nn.PReLU()
    elif name == ActivationTypes.ELU:
        return nn.ELU(inplace=False)
    if name == ActivationTypes.LEAKY_RELU:
        return nn.LeakyReLU(**kwargs)
    elif name == ActivationTypes.SOFTMAX:
        return nn.Softmax()
    elif name == ActivationTypes.IDENTITY:
        return nn.Identity()
    elif name == ActivationTypes.SIN:
        return SinActivation()
    else:
        raise NotImplementedError(f"Activation {name} not implemented.")
