from __future__ import annotations

from typing import Any, List

from cores.core.contracts.logger import Logger
from cores.core.values.constants import Framework


class ListLogger(Logger):
    def __init__(self, logger_list: list[Logger]):
        self.logger_list = logger_list
        super().__init__()

    def track_value(self, key: str, value: Any, step: int = None) -> None:
        if not self.is_enabled:
            return
        for logger in self.logger_list:
            logger.track_value(key, value, step)

    def track_data(self, data: dict[str, Any], step: int = None) -> None:
        if not self.is_enabled:
            return
        for logger in self.logger_list:
            logger.track_data(data, step)

    def track_table(
        self, key: str, columns: List[str], data: List[List[Any]], step: int = None
    ) -> None:
        if not self.is_enabled:
            return
        for logger in self.logger_list:
            logger.track_table(key, columns, data, step)

    def save_model(self, file_name: str, model: Any, framework: str = Framework.PYTORCH) -> None:
        if not self.is_enabled:
            return
        for logger in self.logger_list:
            logger.save_model(file_name=file_name, model=model, framework=framework)
