from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional


class TaskMetricsTorch(ABC):
    def __init__(self, full_names: Optional[List[str]] = None, device: str = "cpu"):
        self.metrics_dict = {}
        self.device = device

        if full_names is not None:
            for full_name in full_names:
                self.add(full_name)

    def get_keys(self, filter: Optional[str] = None) -> List[str]:
        if filter is None:
            return list(self.metrics_dict.keys())
        else:
            return [k for k in self.metrics_dict.keys() if re.search(filter, k) is not None]

    def _get_name(self, full_name: str) -> str:
        if "/" in full_name:
            full_name = full_name.split("/")[-1]

        if "__" in full_name:
            full_name = full_name.split("__")[-1]

        return full_name

    @abstractmethod
    def _add(self, full_name: str, **kwargs) -> None:
        pass

    def add(self, full_name: str, **kwargs) -> None:
        self._add(full_name=full_name, **kwargs)

    def remove(self, full_name: Optional[str] = None) -> None:
        if full_name is None:
            self.metrics_dict = {}
        else:
            del self.metrics_dict[full_name]

    def reset(self, regex: Optional[str] = None) -> None:
        keys_list = self.get_keys(regex)
        for k, v in self.metrics_dict.items():
            if k in keys_list:
                v.reset()

    @abstractmethod
    def _update(self, full_name: str, **kwargs) -> None:
        pass

    def update(self, full_name: str, **kwargs) -> None:
        if full_name not in self.metrics_dict.keys():
            self.add(full_name)
        self._update(full_name=full_name, **kwargs)

    def compute(self, regex: Optional[str] = None) -> dict[str, float]:
        keys_list = self.get_keys(regex)
        metric_dict = {}
        for k in keys_list:
            v = self.metrics_dict[k]
            value = v.compute()
            metric_dict[k] = value
        return metric_dict
