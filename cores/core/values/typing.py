from __future__ import annotations

SplitsList = list


class Dataset:
    def __init__(self):
        pass


class Tensor2D:
    def __init__(self, samples_num: int, features_dim: int):
        self.samples_num = samples_num
        self.features_dim = features_dim


class Tensor1D:
    def __init__(self, samples_num: int):
        self.samples_num = samples_num


class Tensor0D:
    def __init__(self):
        pass


class Tensor:
    def __init__(self):
        pass
