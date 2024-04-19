from __future__ import annotations

import re
from typing import Optional

import torchmetrics as tm


class MetricTrackerTorch:
    def __init__(self):
        self.metric_dict = {}

    def get_keys(self, filter: Optional[str] = None):
        if filter is None:
            return list(self.metric_dict.keys())
        else:
            return [k for k in self.metric_dict.keys() if re.search(filter, k) is not None]

    def add(self, name: str, mode: str = "mean") -> None:
        if mode == "mean":
            metric = tm.MeanMetric()
        else:
            raise ValueError(f"mode {mode} not supported")
        self.metric_dict[name] = metric

    def reset(self, regex: Optional[str] = None) -> None:
        keys_list = self.get_keys(regex)
        for k, v in self.metric_dict.items():
            if k in keys_list:
                v.reset()

    def update(self, name: str, value: float) -> None:
        if name not in self.metric_dict.keys():
            self.add(name)
        self.metric_dict[name].update(value)

    def compute(self, regex: Optional[str] = None) -> float | dict[str, float]:
        keys_list = self.get_keys(regex)

        metric_dict = {}
        for k in keys_list:
            v = self.metric_dict[k]
            value = v.compute()
            metric_dict[k] = value
        return metric_dict
