from __future__ import annotations

import os
from abc import ABC, abstractmethod

from cores.utils.plotter.plotter import Plotter


class EDA(ABC):
    def __init__(self, root: str, plotter: Plotter = None):
        self.root = root

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.plotter = plotter

    @abstractmethod
    def report(self, *args, **kwargs) -> None:
        pass
