from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omegaconf import OmegaConf


class JobsManager(ABC):

    @classmethod
    def from_config_file(cls, config_file: str) -> JobsManager:
        # Load the configuration file
        config = OmegaConf.load(config_file)
        config = OmegaConf.to_object(config)
        return cls(**config)

    @classmethod
    def from_dotlist(cls, config: str) -> JobsManager:
        assert isinstance(config, str)

        config = OmegaConf.from_dotlist(config.split(" "))
        # Convert it to a dictionary
        config = OmegaConf.to_object(config)

        return cls(**config)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> JobsManager:

        return JobsManager(**config)

    @abstractmethod
    def submit(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_identifier(self, i: int) -> str:
        pass

    @abstractmethod
    def add_job(self, command: str, i: int) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass
