from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from cores.core.values.constants import Framework


class Logger(ABC, logging.Logger):
    def __init__(
        self, config: Dict[str, Any] = None, folder: Optional[str] = None, enable: bool = True
    ):
        self.is_enabled = enable
        self.config = config
        self.step = 0
        self._folder = folder

        if self._folder is not None and not os.path.exists(self._folder):
            os.makedirs(self._folder)

        super().__init__("MyLogger")

    def __str__(self):
        return f"{self.__class__.__name__}"

    def folder(self) -> str:
        return self._folder

    def finish(self):
        pass

    def enable(self):
        self.is_enabled = True

    def disable(self):
        self.is_enabled = False

    @abstractmethod
    def track_value(self, key: str, value: Any, step: int = None) -> None:
        pass

    @abstractmethod
    def track_data(self, data: dict[str, Any], step: int = None) -> None:
        pass

    @abstractmethod
    def track_table(
        self, key: str, columns: List[str], data: List[List[Any]], step: int = None
    ) -> None:
        pass

    @abstractmethod
    def save_model(self, file_name: str, model: Any, framework: str = Framework.PYTORCH) -> None:
        pass

    def increment_step(self) -> None:
        self.step += 1


#
