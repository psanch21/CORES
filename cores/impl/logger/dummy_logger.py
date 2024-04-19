from __future__ import annotations

from typing import Any, List

from cores.core.contracts.logger import Logger


class DummyLogger(Logger):
    def track_value(self, key: str, value: Any, step: int = None) -> None:
        pass

    def track_data(self, data: dict[str, Any], step: int = None) -> None:
        pass

    def track_table(
        self, key: str, columns: List[str], data: List[List[Any]], step: int = None
    ) -> None:
        pass

    def save_model(file_name: str, model: Any, framework: str = None) -> None:
        pass
