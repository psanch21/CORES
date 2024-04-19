from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from cores.core.contracts.logger import Logger
from cores.core.values.constants import Framework
from cores.utils.file_io import FileIO


class FSLogger(Logger):
    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        enable: bool = True,
        folder: str = ".logs",
    ):
        super().__init__(config=config, folder=folder, enable=enable)

        # Save config
        if self.config is not None:
            FileIO.dict_to_yaml(
                my_dict=config, file_path=os.path.join(self.folder(), "config.yaml")
            )

    def folder(self) -> str:
        return self._folder

    def safe_file_path(self, file_path: str) -> str:
        folder = os.path.dirname(file_path)

        if not os.path.exists(folder):
            os.makedirs(folder)

        return file_path

    def track_value(self, key: str, value: Any, step: int = None) -> None:
        data = {key: value}

        self.track_data(data=data, step=step)

    def track_data(self, data: dict[str, Any], step: int = None) -> None:
        if not self.is_enabled:
            return

        step = step if step else self.step
        # Add dict to file in a new line
        assert "step" not in data

        for key, value in data.items():
            file_path = self.safe_file_path(os.path.join(self.folder(), f"{key}.txt"))
            if isinstance(value, torch.Tensor):
                value = value.item()

            row = {key: value, "step": step}

            with open(file_path, "a") as f:
                f.write(f"{row}\n")

    def track_table(
        self, key: str, columns: List[str], data: List[List[Any]], step: int = None
    ) -> None:
        if not self.is_enabled:
            return

        step = step if step else self.step

        df = pd.DataFrame(columns=columns, data=data)

        file_path = self.safe_file_path(os.path.join(self.folder(), f"{key}__{step}.csv"))

        df.to_csv(file_path, sep=";")

    def save_model(
        self,
        file_name: str,
        model: Any,
        framework: str = Framework.PYTORCH,
        extra_data: Dict[str, Any] = None,
    ) -> None:
        raise NotImplementedError
