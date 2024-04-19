from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch

import wandb
from cores.core.contracts.logger import Logger
from cores.core.values.constants import Framework
from cores.utils import FileIO


class WandBLogger(Logger):
    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        enable: bool = True,
        **kwargs,
    ):
        super().__init__(config=config, enable=enable)

        assert "dir" in kwargs
        folder = kwargs["dir"]
        if self.is_enabled:
            if wandb.run is None:
                if "key" in kwargs:
                    key = kwargs["key"]
                    del kwargs["key"]
                    wandb.login(key=key)
                self.run = wandb.init(config=config, **kwargs)
            else:
                self.run = wandb.run
            folder = self.run.dir
        super().__init__(config=config, folder=folder, enable=enable)

        if not self.is_enabled:
            # Save config yo taml file

            FileIO.dict_to_yaml(config, os.path.join(self.folder(), "config.yaml"))

    def folder(self) -> str:
        return self._folder

    def track_value(self, key: str, value: Any, step: int = None) -> None:
        if self.is_enabled:
            wandb.log({key: value}, step=step or self.step)

    def track_data(self, data: dict[str, Any], step: int = None) -> None:
        if self.is_enabled:
            wandb.log(data, step=step or self.step)

    def track_table(
        self, key: str, columns: List[str], data: List[List[Any]], step: int = None
    ) -> None:
        if self.is_enabled:
            data_ = []
            for row in data:
                row_data = []
                for elem in row:
                    if isinstance(elem, str) and ".png" in elem:
                        elem = wandb.Image(elem)
                    row_data.append(elem)
                data_.append(row_data)

            wandb.log({key: wandb.Table(columns=columns, data=data_)}, step=step or self.step)

    def save_model(
        self,
        file_name: str,
        model: Any,
        framework: str = Framework.PYTORCH,
        extra_data: Dict[str, Any] = None,
    ) -> None:
        if not self.is_enabled:
            return

        if framework == Framework.PYTORCH:
            file_path = os.path.join(wandb.run.dir, file_name)
            logging.info(f"Saving model to {file_path}")
            data = {}
            data["model_state_dict"] = model.state_dict()

            if extra_data is not None:
                data.update(extra_data)
            torch.save(data, file_path)

    def finish(self):
        if self.is_enabled:
            self.run.finish()
